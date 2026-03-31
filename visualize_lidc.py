"""
visualize_lidc.py  —  Prototype visualisation for LIDC binary classification
=============================================================================
For each prototype, finds the K most-activating patches in the validation set
and saves:
  - The original nodule image with the prototype region highlighted.
  - A colour heatmap overlay.
  - A summary grid per class.

Usage:
    python visualize_lidc.py \
        --data_path  datasets/LIDC \
        --model_path output/LIDC/run1/checkpoints/best-auc.pth \
        --output_dir output/LIDC/vis \
        --base_architecture deit_tiny_patch16_224 \
        --prototype_shape 20 192 1 1 \
        --use_global True \
        --top_k 5
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import protopformer
from tools.datasets import build_dataset, LIDC_MEAN, LIDC_STD
from tools.utils    import str2bool

CLASSES = ['benign', 'malignant']


# ── helpers ────────────────────────────────────────────────────────────────────

def denormalise(tensor, mean=LIDC_MEAN, std=LIDC_STD):
    """Undo ImageNet normalisation for display."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (tensor * s + m).clamp(0, 1)


def tensor_to_pil(t):
    arr = (denormalise(t).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def activation_to_heatmap(act_map, img_size):
    """
    act_map: (H, W) numpy array of activation values.
    Returns a PIL image (heatmap) at img_size × img_size.
    """
    act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
    act_up  = F.interpolate(
        torch.tensor(act_map)[None, None],
        size=(img_size, img_size), mode='bilinear', align_corners=False
    )[0, 0].numpy()
    cm = plt.get_cmap('jet')
    return Image.fromarray((cm(act_up)[:, :, :3] * 255).astype(np.uint8))


def overlay(img_pil, heatmap_pil, alpha=0.45):
    return Image.blend(img_pil.convert('RGB'),
                       heatmap_pil.convert('RGB'), alpha)


# ── main visualisation ─────────────────────────────────────────────────────────

@torch.no_grad()
def visualise(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load model ─────────────────────────────────────────────────────────────
    model = protopformer.construct_PPNet(
        base_architecture           = args.base_architecture,
        pretrained                  = False,
        img_size                    = args.img_size,
        prototype_shape             = args.prototype_shape,
        num_classes                 = 2,
        reserve_layers              = args.reserve_layers,
        reserve_token_nums          = args.reserve_token_nums,
        use_global                  = args.use_global,
        global_proto_per_class      = args.global_proto_per_class,
        prototype_activation_function = 'log',
        add_on_layers_type          = 'regular',
    )
    ckpt = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    num_prototypes = model.num_prototypes
    num_per_class  = model.num_prototypes_per_class

    # ── dataset ────────────────────────────────────────────────────────────────
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # raw images (no normalisation) for display
    raw_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
    ])

    # ── collect activations ────────────────────────────────────────────────────
    # For each prototype store top-K (activation_value, image_idx, act_map)
    topk = args.top_k
    best = {p: [] for p in range(num_prototypes)}  # list of (score, img_idx, act_map_np)

    for img_idx in range(len(dataset_val)):
        img_t, label = dataset_val[img_idx]
        img_batch = img_t.unsqueeze(0).to(device)

        cls_token_attn, proto_acts = model.push_forward(img_batch)
        # proto_acts: (1, num_prototypes, fea_h, fea_w)
        proto_acts = proto_acts[0].cpu()   # (num_prototypes, H, W)

        for p in range(num_prototypes):
            act_map   = proto_acts[p].numpy()       # (H, W)
            score     = float(act_map.max())
            best[p].append((score, img_idx, act_map))

        if img_idx % 100 == 0:
            print(f"  processed {img_idx}/{len(dataset_val)}")

    # keep only top-K per prototype
    for p in range(num_prototypes):
        best[p].sort(key=lambda x: x[0], reverse=True)
        best[p] = best[p][:topk]

    # ── save images ────────────────────────────────────────────────────────────
    for p in range(num_prototypes):
        class_idx  = p // num_per_class
        class_name = CLASSES[class_idx]
        proto_dir  = os.path.join(args.output_dir, f'proto_{p:03d}_{class_name}')
        os.makedirs(proto_dir, exist_ok=True)

        for rank, (score, img_idx, act_map) in enumerate(best[p]):
            raw_img, true_label = dataset_val[img_idx]
            pil_img   = tensor_to_pil(raw_img)
            heatmap   = activation_to_heatmap(act_map, args.img_size)
            overlaid  = overlay(pil_img, heatmap)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(pil_img, cmap='gray');    axes[0].set_title('Original')
            axes[1].imshow(heatmap);                 axes[1].set_title('Activation')
            axes[2].imshow(overlaid);                axes[2].set_title('Overlay')
            for ax in axes:
                ax.axis('off')

            true_name = CLASSES[true_label]
            fig.suptitle(
                f'Proto {p} ({class_name}) | Rank {rank+1} | Score {score:.3f}'
                f' | True: {true_name}',
                fontsize=11
            )
            fig.tight_layout()
            fig.savefig(os.path.join(proto_dir, f'rank{rank+1:02d}.png'),
                        dpi=100, bbox_inches='tight')
            plt.close(fig)

    # ── summary grid per class ────────────────────────────────────────────────
    for class_idx, class_name in enumerate(CLASSES):
        protos_for_class = [p for p in range(num_prototypes)
                            if p // num_per_class == class_idx]
        fig, axes = plt.subplots(len(protos_for_class), topk,
                                  figsize=(topk * 3, len(protos_for_class) * 3))
        if len(protos_for_class) == 1:
            axes = [axes]

        for row, p in enumerate(protos_for_class):
            for col, (score, img_idx, act_map) in enumerate(best[p]):
                raw_img, _ = dataset_val[img_idx]
                pil_img    = tensor_to_pil(raw_img)
                heatmap    = activation_to_heatmap(act_map, args.img_size)
                overlaid   = overlay(pil_img, heatmap)
                axes[row][col].imshow(overlaid)
                axes[row][col].set_title(f'P{p} R{col+1}\n{score:.2f}',
                                          fontsize=8)
                axes[row][col].axis('off')

        fig.suptitle(f'Top-{topk} patches per prototype  |  Class: {class_name}',
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'summary_{class_name}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved summary grid: summary_{class_name}.png")

    print(f"\nDone. All visualisations saved to: {args.output_dir}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser('ProtoPFormer LIDC visualisation')
    p.add_argument('--data_path',          type=str, required=True)
    p.add_argument('--lidc_format',        type=str, default='folder')
    p.add_argument('--model_path',         type=str, required=True)
    p.add_argument('--output_dir',         type=str, required=True)
    p.add_argument('--base_architecture',  type=str,
                   default='deit_tiny_patch16_224')
    p.add_argument('--prototype_shape',    nargs='+', type=int,
                   default=[20, 192, 1, 1])
    p.add_argument('--img_size',           type=int, default=224)
    p.add_argument('--use_global',         type=str2bool, default=True)
    p.add_argument('--global_proto_per_class', type=int, default=5)
    p.add_argument('--reserve_layers',     nargs='+', type=int, default=[11])
    p.add_argument('--reserve_token_nums', nargs='+', type=int, default=[81])
    p.add_argument('--top_k',              type=int,  default=5,
                   help='Top-K most activating patches per prototype')
    args = p.parse_args()
    visualise(args)
