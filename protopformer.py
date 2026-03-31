"""
protopformer.py  —  adapted for LIDC-IDRI lung-nodule binary classification
=============================================================================
Key changes from the original:
  1. num_classes fixed to 2 (benign / malignant).
  2. prototype_shape default reduced to fit a 2-class problem:
       (20, 192, 1, 1)  →  10 local prototypes per class
  3. global_proto_per_class default reduced to 5.
  4. class-identity matrices and last-layer initialisation work for 2 classes.
  5. No other architectural changes — the dual-branch design and PPC loss are
     preserved exactly because they are the interpretability core.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.deit_features import deit_tiny_patch_features, deit_small_patch_features
from tools.cait_features  import cait_xxs24_224_features

base_architecture_to_features = {
    'deit_tiny_patch16_224':  deit_tiny_patch_features,
    'deit_small_patch16_224': deit_small_patch_features,
    'cait_xxs24_224':         cait_xxs24_224_features,
}


class PPNet(nn.Module):
    """
    Prototypical Part Network built on a Vision Transformer backbone.

    For LIDC binary classification the default prototype layout is:
        num_classes            = 2
        num_prototypes         = 20   (10 local parts per class)
        global_proto_per_class = 5    (5 global prototypes per class)
    """

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes,
                 reserve_layers=(),
                 reserve_token_nums=(),
                 use_global=False,
                 use_ppc_loss=False,
                 ppc_cov_thresh=2.,
                 ppc_mean_thresh=2,
                 global_coe=0.3,
                 global_proto_per_class=5,
                 init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):
        super().__init__()

        assert num_classes == 2, \
            "This LIDC adaptation is hard-coded for binary (benign/malignant) classification."

        self.img_size           = img_size
        self.prototype_shape    = prototype_shape
        self.num_prototypes     = prototype_shape[0]
        self.num_classes        = num_classes
        self.reserve_layers     = list(reserve_layers)
        self.reserve_token_nums = list(reserve_token_nums)
        self.use_global         = use_global
        self.use_ppc_loss       = use_ppc_loss
        self.ppc_cov_thresh     = ppc_cov_thresh
        self.ppc_mean_thresh    = ppc_mean_thresh
        self.global_coe         = global_coe
        self.global_proto_per_class = global_proto_per_class
        self.epsilon            = 1e-4

        self.reserve_layer_nums = list(zip(self.reserve_layers,
                                           self.reserve_token_nums))

        self.num_prototypes_global  = self.num_classes * self.global_proto_per_class
        self.prototype_shape_global = ([self.num_prototypes_global]
                                       + list(self.prototype_shape[1:]))

        self.prototype_activation_function = prototype_activation_function

        assert self.num_prototypes % self.num_classes == 0, \
            "num_prototypes must be divisible by num_classes (2)."

        # ── class-identity matrices ───────────────────────────────────────────
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                     self.num_classes)
        self.prototype_class_identity_global = torch.zeros(
            self.num_prototypes_global, self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        num_prototypes_per_class_global = (self.num_prototypes_global
                                           // self.num_classes)
        for j in range(self.num_prototypes_global):
            self.prototype_class_identity_global[
                j, j // num_prototypes_per_class_global] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # ── backbone ──────────────────────────────────────────────────────────
        self.features = features
        features_name = str(self.features).upper()
        if features_name.startswith('MYVISION') or features_name.startswith('MYCAIT'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules()
                 if isinstance(i, nn.Linear)][-1].out_features
        else:
            raise Exception('Unsupported base architecture.')

        self.num_patches = self.features.patch_embed.num_patches

        # ── add-on projection layers ──────────────────────────────────────────
        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) \
                    or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1],
                                           current_in_channels // 2)
                add_on_layers += [
                    nn.Conv2d(current_in_channels, current_out_channels,
                              kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(current_out_channels, current_out_channels,
                              kernel_size=1),
                ]
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(first_add_on_layer_in_channels,
                          self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
            )

        # ── prototype vectors ─────────────────────────────────────────────────
        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True)
        if self.use_global:
            self.prototype_vectors_global = nn.Parameter(
                torch.rand(self.prototype_shape_global), requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        # ── classification heads ──────────────────────────────────────────────
        # For binary classification the FC maps prototype scores → 2 logits.
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False)
        self.last_layer_global = nn.Linear(self.num_prototypes_global,
                                           self.num_classes, bias=False)
        self.last_layer.weight.requires_grad        = False
        self.last_layer_global.weight.requires_grad = False

        self.all_attn_mask  = None
        self.teacher_model  = None
        self.scale          = self.prototype_shape[1] ** -0.5

        if init_weights:
            self._initialize_weights()

    # ── feature extraction ─────────────────────────────────────────────────────

    def conv_features(self, x, reserve_layer_nums=()):
        feature_module_name = self.features.__class__.__name__
        if 'Vision' in feature_module_name or 'MyCait' in feature_module_name:
            if self.use_global:
                cls_embed, x_embed = \
                    self.features.forward_feature_patch_embed_all(x)
            else:
                x_embed = self.features.forward_feature_patch_embed(x)
                cls_embed = None

            fea_size = int(x_embed.shape[1] ** 0.5)
            dim      = x_embed.shape[-1]

            token_attn = None
            x_out, (cls_token_attn, _) = \
                self.features.forward_feature_mask_train_direct(
                    cls_embed, x_embed, token_attn, reserve_layer_nums)

            final_reserve_num     = reserve_layer_nums[-1][1]
            final_reserve_indices = torch.topk(cls_token_attn,
                                               k=final_reserve_num, dim=-1)[1]
            final_reserve_indices = final_reserve_indices.sort(dim=-1)[0]
            final_reserve_indices = final_reserve_indices[:, :, None].repeat(
                1, 1, dim)

            cls_tokens, img_tokens = x_out[:, :1], x_out[:, 1:]
            img_tokens = torch.gather(img_tokens, 1, final_reserve_indices)

            B   = img_tokens.shape[0]
            fea_len = img_tokens.shape[1]
            fea_hw  = int(fea_len ** 0.5)

            cls_tokens = cls_tokens.permute(0, 2, 1).reshape(B, dim, 1, 1)
            img_tokens = img_tokens.permute(0, 2, 1).reshape(
                B, dim, fea_hw, fea_hw)
        else:
            x_out = self.features(x)
            cls_tokens = x_out
            img_tokens = x_out

        cls_tokens = self.add_on_layers(cls_tokens)
        img_tokens = self.add_on_layers(img_tokens)
        return (cls_tokens, img_tokens), (token_attn, cls_token_attn, None)

    # ── distance / similarity ──────────────────────────────────────────────────

    def _l2_convolution_single(self, x, prototype_vectors):
        temp_ones = torch.ones(prototype_vectors.shape, device=x.device)
        x2          = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=temp_ones)
        p2          = prototype_vectors ** 2
        p2          = torch.sum(p2, dim=(1, 2, 3)).view(-1, 1, 1)
        xp          = F.conv2d(input=x, weight=prototype_vectors)
        distances   = F.relu(x2_patch_sum - 2 * xp + p2)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def prototype_distances(self, x, reserve_layer_nums=()):
        (cls_tokens, img_tokens), auxi = self.conv_features(x, reserve_layer_nums)
        return (cls_tokens, img_tokens), auxi

    def get_activations(self, tokens, prototype_vectors):
        B = tokens.shape[0]
        num_proto = prototype_vectors.shape[0]
        distances   = self._l2_convolution_single(tokens, prototype_vectors)
        activations = self.distance_2_similarity(distances)
        total_proto_act = activations
        fea_size = activations.shape[-1]
        if fea_size > 1:
            activations = F.max_pool2d(activations,
                                       kernel_size=(fea_size, fea_size))
        activations = activations.reshape(B, num_proto)
        if self.use_global:
            return activations, (distances, total_proto_act)
        return activations

    # ── PPC loss ───────────────────────────────────────────────────────────────

    def batch_cov(self, points, weights):
        B, N, D = points.size()
        weights  = weights / weights.sum(dim=-1, keepdim=True) * N
        mean     = (points * weights[:, :, None]).mean(dim=1).unsqueeze(1)
        diffs    = (points - mean).reshape(B * N, D)
        prods    = torch.bmm(diffs.unsqueeze(2),
                             diffs.unsqueeze(1)).reshape(B, N, D, D)
        prods    = prods * weights[:, :, None, None]
        bcov     = prods.sum(dim=1) / (N - 1)
        return mean, bcov

    def get_PPC_loss(self, total_proto_act, cls_attn_rollout,
                     original_fea_len, label):
        B              = total_proto_act.shape[0]
        original_fea_size = int(original_fea_len ** 0.5)
        proto_per_class = self.num_prototypes_per_class

        discrete_values = torch.FloatTensor(
            [[x, y]
             for x in range(original_fea_size)
             for y in range(original_fea_size)]
        ).to(total_proto_act.device)
        discrete_values = discrete_values[None].repeat(
            B * proto_per_class, 1, 1)

        discrete_weights   = torch.zeros(B, proto_per_class,
                                         original_fea_len,
                                         device=total_proto_act.device)
        total_proto_act    = total_proto_act.flatten(start_dim=2)
        final_token_num    = total_proto_act.shape[-1]

        proto_indices = (label * proto_per_class).unsqueeze(-1).repeat(
            1, proto_per_class)
        proto_indices += torch.arange(proto_per_class,
                                      device=total_proto_act.device)
        proto_indices = proto_indices[:, :, None].repeat(1, 1, final_token_num)
        total_proto_act = torch.gather(total_proto_act, 1, proto_indices)

        reserve_token_indices = torch.topk(
            cls_attn_rollout, k=final_token_num, dim=-1)[1].sort(dim=-1)[0]
        reserve_token_indices = reserve_token_indices[:, None, :].repeat(
            1, proto_per_class, 1)
        discrete_weights.scatter_(2, reserve_token_indices, total_proto_act)
        discrete_weights = discrete_weights.reshape(B * proto_per_class, -1)

        mean_ma, cov_ma = self.batch_cov(discrete_values, discrete_weights)

        ppc_cov_loss = (cov_ma[:, 0, 0] + cov_ma[:, 1, 1]) / 2
        ppc_cov_loss = F.relu(ppc_cov_loss - self.ppc_cov_thresh).mean()

        mean_ma   = mean_ma.reshape(B, proto_per_class, 2)
        mean_diff = torch.cdist(mean_ma, mean_ma)
        mean_mask = (1. - torch.eye(proto_per_class,
                                    device=mean_diff.device))
        ppc_mean_loss = F.relu(
            (self.ppc_mean_thresh - mean_diff) * mean_mask).mean()

        return ppc_cov_loss, ppc_mean_loss

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, x):
        reserve_layer_nums = self.reserve_layer_nums

        if not self.training:
            (cls_tokens, img_tokens), (_, cls_token_attn, _) = \
                self.prototype_distances(x, reserve_layer_nums)
            global_activations, _ = self.get_activations(
                cls_tokens, self.prototype_vectors_global)
            local_activations, (distances, _) = self.get_activations(
                img_tokens, self.prototype_vectors)
            logits_global = self.last_layer_global(global_activations)
            logits_local  = self.last_layer(local_activations)
            logits = (self.global_coe * logits_global
                      + (1. - self.global_coe) * logits_local)
            return logits, (cls_token_attn, distances,
                            logits_global, logits_local)

        # ── training forward ──────────────────────────────────────────────────
        (cls_tokens, img_tokens), (_, cls_attn_rollout, _) = \
            self.prototype_distances(x, reserve_layer_nums)
        cls_attn_rollout = cls_attn_rollout.detach()

        B            = cls_tokens.shape[0]
        original_fea_size = int(cls_attn_rollout.shape[-1] ** 0.5)

        global_activations, _ = self.get_activations(
            cls_tokens, self.prototype_vectors_global)
        local_activations, (_, total_proto_act) = self.get_activations(
            img_tokens, self.prototype_vectors)

        logits_global = self.last_layer_global(global_activations)
        logits_local  = self.last_layer(local_activations)
        logits = (self.global_coe * logits_global
                  + (1. - self.global_coe) * logits_local)

        original_fea_len = original_fea_size ** 2
        return logits, (None, torch.zeros(1, device=logits.device),
                        total_proto_act, cls_attn_rollout, original_fea_len)

    def push_forward(self, x):
        reserve_layer_nums = self.reserve_layer_nums
        (cls_tokens, img_tokens), (_, cls_token_attn, _) = \
            self.prototype_distances(x, reserve_layer_nums)
        _, (_, proto_acts) = self.get_activations(
            img_tokens, self.prototype_vectors)
        return cls_token_attn, proto_acts

    # ── weight initialisation ──────────────────────────────────────────────────

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        pos = torch.t(self.prototype_class_identity)
        neg = 1 - pos
        self.last_layer.weight.data.copy_(
            1 * pos + incorrect_strength * neg)
        if hasattr(self, 'last_layer_global'):
            pos_g = torch.t(self.prototype_class_identity_global)
            neg_g = 1 - pos_g
            self.last_layer_global.weight.data.copy_(
                1 * pos_g + incorrect_strength * neg_g)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


# ── constructor ────────────────────────────────────────────────────────────────

def construct_PPNet(base_architecture,
                    pretrained=True,
                    img_size=224,
                    prototype_shape=(20, 192, 1, 1),   # 10 per class × 2 classes
                    num_classes=2,
                    reserve_layers=(),
                    reserve_token_nums=(),
                    use_global=False,
                    use_ppc_loss=False,
                    ppc_cov_thresh=1.,
                    ppc_mean_thresh=2.,
                    global_coe=0.5,
                    global_proto_per_class=5,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):

    assert num_classes == 2, "LIDC adaptation expects num_classes=2."

    features = base_architecture_to_features[base_architecture](
        pretrained=pretrained)
    proto_layer_rf_info = [14, 16, 16, 8.0]   # valid for 224px DeiT/CaiT

    return PPNet(
        features=features,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=proto_layer_rf_info,
        num_classes=num_classes,
        reserve_layers=reserve_layers,
        reserve_token_nums=reserve_token_nums,
        use_global=use_global,
        use_ppc_loss=use_ppc_loss,
        ppc_cov_thresh=ppc_cov_thresh,
        ppc_mean_thresh=ppc_mean_thresh,
        global_coe=global_coe,
        global_proto_per_class=global_proto_per_class,
        init_weights=True,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type,
    )
