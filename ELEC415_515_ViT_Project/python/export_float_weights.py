"""Export the checkpoint to weights_float/, one .npy file per tensor ('.' in names becomes '__').

weights_float/source.txt notes which checkpoint was exported (name and content hash); the
later steps report it. The folder is replaced as a whole, so no tensor of an earlier export
survives.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from model import build_mnist_vit
from vit_fixed import ROOT, checkpoint_id, find_input


def main():
    parser = argparse.ArgumentParser(description="Export the trained PyTorch checkpoint to one .npy file per tensor.")
    parser.add_argument("--checkpoint", type=str, default="ViT_model.pt", help="a name in checkpoint/ or a path")
    parser.add_argument("--out_dir", type=str, default=str(ROOT / "weights_float"))
    args = parser.parse_args()

    checkpoint = find_input(args.checkpoint, "checkpoint", ".pt")
    state = torch.load(checkpoint, map_location="cpu")
    # Checkpoints saved from a wrapped model (DataParallel, torch.compile) prefix every name.
    for prefix in ("module.", "_orig_mod."):
        if all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    expected = set(build_mnist_vit().state_dict())
    if set(state) != expected:
        missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
        raise SystemExit(f"{checkpoint} does not match the model in model.py: "
                         f"missing {missing[:3]}{' ...' if len(missing) > 3 else ''}, "
                         f"unexpected {extra[:3]}{' ...' if len(extra) > 3 else ''}")

    out_dir = Path(args.out_dir)
    tmp_dir = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    for name, tensor in state.items():
        np.save(tmp_dir / f"{name.replace('.', '__')}.npy", tensor.detach().cpu().numpy())
    (tmp_dir / "source.txt").write_text(checkpoint_id(checkpoint) + "\n")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    print(f"Exported {len(state)} tensors of {checkpoint} to {out_dir}")


if __name__ == "__main__":
    main()
