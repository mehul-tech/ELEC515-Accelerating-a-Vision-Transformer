"""Attention-head importance and greedy head pruning, following Michel et al., 2019,
"Are Sixteen Heads Really Better than One?".

Every head h of block l gets a gate xi[l, h] that multiplies its output (model.set_head_mask).
The importance of a head is how strongly the loss reacts to its gate:
    I[l, h] = mean over training batches of |dLoss / dxi[l, h]|,  with every gate at 1.
The scores of each block are L2-normalized so that blocks can be compared. Heads are then
masked one at a time, least important first, and the test accuracy is measured after every
step. The mask with --prune heads removed is written to head_masks/prune<N>.txt, ready for
export_rtl_data.py and the hardware.
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from data import evaluate, get_loaders
from model import build_mnist_vit
from vit_fixed import ROOT, find_input
N_LAYERS, N_HEADS = 6, 4


# Returns the raw and the per-block normalized importance, both N_LAYERS x N_HEADS.
def head_importance(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    # One gate per head, all at 1 (= keep every head); requires_grad lets us ask for dLoss/dgate.
    mask = torch.ones(N_LAYERS, N_HEADS, device=device, requires_grad=True)
    model.set_head_mask(mask)
    importance = torch.zeros(N_LAYERS, N_HEADS, device=device)
    batches = 0
    model.eval()
    for x, y in loader:
        # TODO (1): accumulate the gradient magnitude of the loss w.r.t. the head gates
        # HINT: compute the cross-entropy loss of this batch (move x and y to `device` first),
        # HINT: take its gradient with respect to `mask` (torch.autograd.grad(loss, mask)
        # HINT: returns a tuple), and add the absolute value to `importance`. One line for the
        # HINT: loss, one for the accumulation.
        loss = criterion(model(x.to(device)), y.to(device))
        importance += torch.autograd.grad(loss, mask)[0].abs().detach()
        batches += 1
    model.set_head_mask(None)
    importance = importance / batches
    # TODO (2): L2-normalize the importance scores within each block
    # HINT: divide every row (one block) of `importance` by its L2 norm, so the four scores of
    # HINT: a block have unit length, and store the result in `normalized` (same shape).
    # HINT: Guard against division by zero (e.g. clamp_min(1e-12)). One line.
    normalized = importance / torch.linalg.vector_norm(
        importance, dim=1, keepdim=True
    ).clamp_min(1e-12)
    return importance.cpu(), normalized.cpu()


# Same format as head_masks/*.txt: one row per block, 1 = keep.
def write_mask(path, mask):
    rows = [" ".join(str(int(v)) for v in row) for row in mask.tolist()]
    Path(path).write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Gradient-based attention-head importance and greedy head pruning (Michel et al., 2019).")
    parser.add_argument("--checkpoint", type=str, default="ViT_model.pt", help="a name in checkpoint/ or a path")
    parser.add_argument("--prune", type=int, default=4, help="number of heads removed in the exported mask")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.prune <= N_LAYERS * N_HEADS:
        raise SystemExit(f"--prune must be between 1 and {N_LAYERS * N_HEADS}")

    # The importance is averaged over shuffled, randomly shifted training batches; a fixed seed
    # makes the ranking reproducible from run to run.
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(args.batch_size, 0)
    model = build_mnist_vit()
    model.load_state_dict(torch.load(find_input(args.checkpoint, "checkpoint", ".pt"), map_location="cpu"))
    model.to(device)

    baseline = evaluate(model, test_loader, device)
    raw, normalized = head_importance(model, train_loader, device)
    # TODO (3): rank the 24 heads from least to most important
    # HINT: build a list `ranking` of tuples (normalized score as a float, block index l, head
    # HINT: index h) for all 6 x 4 heads, sorted in increasing order of the score; the loop below
    # HINT: unpacks it as (_, layer, head). The ranking is computed once, before pruning.
    ranking = sorted(
        (float(normalized[l, h]), l, h)
        for l in range(N_LAYERS)
        for h in range(N_HEADS)
    )

    # Mask heads one at a time in ranking order and measure the accuracy after each step.
    mask = torch.ones(N_LAYERS, N_HEADS)
    curve = []
    for step, (_, layer, head) in enumerate(ranking, start=1):
        mask[layer, head] = 0
        model.set_head_mask(mask.to(device))
        acc = evaluate(model, test_loader, device)
        curve.append((step, layer, head, acc))
        print(f"masked {step:2d} heads (+ block {layer + 1} head {head}): test acc {acc:.2%}", flush=True)
        if step == args.prune:
            write_mask(ROOT / "head_masks" / f"prune{args.prune}.txt", mask)
    model.set_head_mask(None)

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    with (results / "head_importance.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block", "head", "raw_importance", "normalized_importance"])
        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                writer.writerow([l + 1, h, f"{float(raw[l, h]):.6e}", f"{float(normalized[l, h]):.6f}"])
    with (results / "head_pruning.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["masked_heads", "block", "head", "test_acc"])
        writer.writerow([0, "", "", f"{baseline:.4f}"])
        for step, layer, head, acc in curve:
            writer.writerow([step, layer + 1, head, f"{acc:.4f}"])

    print(f"baseline test acc: {baseline:.2%}")
    print(f"mask with {args.prune} heads removed written to head_masks/prune{args.prune}.txt")


if __name__ == "__main__":
    main()
