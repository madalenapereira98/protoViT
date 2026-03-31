"""
engine_proto.py  —  training / evaluation loop for LIDC binary classification
==============================================================================
Key differences from the original:
  1. Class-weighted cross-entropy to handle benign/malignant imbalance.
  2. Logs AUC, sensitivity, specificity alongside accuracy.
  3. PPC loss wiring unchanged — works for 2 classes.
"""

import math
import torch
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                              balanced_accuracy_score)
import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────────

def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def compute_class_weights(dataset, device):
    """
    Inverse-frequency weighting for the two classes.
    Pass your LIDCFolderDataset / LIDCCSVDataset instance.
    """
    labels = [lbl for _, lbl in dataset.samples]
    n_total   = len(labels)
    n_benign  = labels.count(0)
    n_malignant = labels.count(1)
    w_benign    = n_total / (2.0 * n_benign)    if n_benign    > 0 else 1.0
    w_malignant = n_total / (2.0 * n_malignant) if n_malignant > 0 else 1.0
    print(f"Class weights  benign={w_benign:.3f}  malignant={w_malignant:.3f}")
    return torch.tensor([w_benign, w_malignant], dtype=torch.float).to(device)


# ── one training epoch ─────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    loss_scaler, max_norm=None, model_ema=None, mixup_fn=None,
                    args=None, tb_writer=None, iteration=0):
    model.train()

    total_loss = 0.0
    correct    = 0
    total      = 0

    ppc_cov_coe  = getattr(args, 'ppc_cov_coe',  0.1)
    ppc_mean_coe = getattr(args, 'ppc_mean_coe', 0.5)
    coefs_clst   = getattr(args, 'coefs_clst',   0.8)
    coefs_sep    = getattr(args, 'coefs_sep',    -0.08)
    coefs_l1     = getattr(args, 'coefs_l1',     1e-4)

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(loss_scaler is not None)):
            output = model(images)

            if isinstance(output, tuple):
                logits, aux = output
                # aux = (student_token_attn, attn_loss, total_proto_act,
                #         cls_attn_rollout, original_fea_len)
                student_token_attn = aux[0]
                total_proto_act    = aux[2]
                cls_attn_rollout   = aux[3]
                original_fea_len   = aux[4]
            else:
                logits = output

            ce_loss = criterion(logits, labels)

            # ── PPC loss ──────────────────────────────────────────────────────
            ppc_loss = torch.zeros(1, device=device)
            if (getattr(args, 'use_ppc_loss', False)
                    and total_proto_act is not None
                    and cls_attn_rollout is not None):
                try:
                    ppc_cov_loss, ppc_mean_loss = model.module.get_PPC_loss(
                        total_proto_act, cls_attn_rollout,
                        original_fea_len, labels
                    ) if hasattr(model, 'module') else \
                        model.get_PPC_loss(
                            total_proto_act, cls_attn_rollout,
                            original_fea_len, labels)
                    ppc_loss = (ppc_cov_coe * ppc_cov_loss
                                + ppc_mean_coe * ppc_mean_loss)
                except Exception:
                    pass   # skip PPC if shapes don't align (e.g. tiny batch)

            loss = ce_loss + ppc_loss

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer,
                        clip_value=max_norm,
                        parameters=model.parameters())
        else:
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        it = iteration + batch_idx
        if tb_writer is not None and it % 50 == 0:
            tb_writer.add_scalar('train/ce_loss',  ce_loss.item(),  it)
            tb_writer.add_scalar('train/ppc_loss', ppc_loss.item(), it)
            tb_writer.add_scalar('train/loss',     loss.item(),     it)

    avg_loss = total_loss / total
    acc      = correct / total * 100
    return {'loss': avg_loss, 'acc1': acc}


# ── evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    model.eval()

    all_labels  = []
    all_probs   = []   # P(malignant) — for AUC
    all_preds   = []
    total_loss  = 0.0
    total       = 0

    criterion = torch.nn.CrossEntropyLoss()

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(images)
        logits = output[0] if isinstance(output, tuple) else output

        loss        = criterion(logits, labels)
        probs       = F.softmax(logits, dim=1)[:, 1]   # P(malignant)
        preds       = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total      += images.size(0)

        all_labels.extend(_to_numpy(labels).tolist())
        all_probs.extend(_to_numpy(probs).tolist())
        all_preds.extend(_to_numpy(preds).tolist())

    # ── metrics ───────────────────────────────────────────────────────────────
    acc   = sum(p == l for p, l in zip(all_preds, all_labels)) / total * 100
    avg_loss = total_loss / total

    try:
        auc = roc_auc_score(all_labels, all_probs) * 100
    except ValueError:
        auc = 0.0   # only one class present in batch (shouldn't happen)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn + 1e-8) * 100   # recall for malignant
    specificity = tn / (tn + fp + 1e-8) * 100   # recall for benign
    bal_acc     = balanced_accuracy_score(all_labels, all_preds) * 100

    # acc5 kept for API compatibility with the original evaluate() caller
    return {
        'loss':         avg_loss,
        'acc1':         acc,
        'acc5':         acc,          # meaningless for binary but keeps API
        'auc':          auc,
        'sensitivity':  sensitivity,
        'specificity':  specificity,
        'balanced_acc': bal_acc,
        'global_acc1':  acc,          # placeholder — split logged in main
        'local_acc1':   acc,
    }
