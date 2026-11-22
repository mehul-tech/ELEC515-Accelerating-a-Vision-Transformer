"""Fine-tune the full-precision model with a head mask applied (no quantization).

Pruning heads lowers accuracy; a few epochs of fine-tuning with the pruned heads switched off
lets the remaining heads compensate. The fine-tuned weights are saved as a normal checkpoint.
Afterwards export them and regenerate the hardware data:
  python finetune.py --mask prune4
  python export_float_weights.py --checkpoint ViT_model_finetuned.pt
  python export_rtl_data.py --samples 20
"""

import argparse

import torch
import torch.nn as nn
from torch import optim

from data import evaluate, get_loaders
from model import build_mnist_vit
from vit_fixed import find_input, output_path, read_head_mask


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the model with a head mask applied.")
    parser.add_argument("--mask", type=str, required=True, help="head mask: a name in head_masks/ (e.g. prune4) or a path")
    parser.add_argument("--checkpoint", type=str, default="ViT_model.pt", help="a name in checkpoint/ or a path")
    parser.add_argument("--out_checkpoint", type=str, default="ViT_model_finetuned.pt",
                        help="output: a file name (saved in checkpoint/) or a path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(args.batch_size, 0)

    model = build_mnist_vit()
    model.load_state_dict(torch.load(find_input(args.checkpoint, "checkpoint", ".pt"), map_location="cpu"))
    model.to(device)
    mask_file = find_input(args.mask, "head_masks", ".txt")
    model.set_head_mask(torch.tensor(read_head_mask(mask_file), dtype=torch.float32, device=device))
    print(f"head mask: {mask_file}")
    print(f"test accuracy with the mask, before fine-tuning: {evaluate(model, test_loader, device):.2%}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    for epoch in range(args.epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            loss = criterion(model(x.to(device)), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 50 == 0:
                print(f"epoch {epoch + 1}/{args.epochs} iter {i + 1}/{len(train_loader)} loss {loss.item():.4f}", flush=True)
        print(f"epoch {epoch + 1}: test accuracy with the mask {evaluate(model, test_loader, device):.2%}", flush=True)

    # The mask is not part of the state dict; it is applied again by the hardware.
    out = output_path(args.out_checkpoint, "checkpoint", ".pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out)
    print(f"checkpoint written to {out}")


if __name__ == "__main__":
    main()
