"""Export Q4.12 weights, the first N test images and the golden outputs for the testbench.

weights/        one integer per line, matrices row-major [out][in]
data/           embedding_input.txt (one image per line), label.txt, sample_count.txt
data/<case>/    golden_logits.txt, golden_pred.txt for each mask in head_masks/
With --int8 [SCALES] it also writes the parameters of the LINEAR_INT8 build and its golden
outputs. SCALES is the scales file written by quantize_ptq.py: a name in quant/ or a path;
without a value it is quant/scales_ptq.npz.
quant/blockN/   <layer>_w8.txt, <layer>_x_mult.txt, <layer>_y_mult.txt for wq, wk, wv, fc1, fc2
data/int8/<case>/  golden outputs of the INT8 model for each mask
data/source.txt and data/int8/source.txt note which checkpoint (and scales) the data describe;
run_sim.sh prints them. They are removed when an export starts and written when it has
finished, so run_sim.sh can refuse the data of an interrupted export. The INT8 build also reads
weights/ (biases, LayerNorm, classifier), so an export without --int8 deletes data/int8/: its
golden outputs no longer match weights/.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from vit_fixed import (
    BLOCKS, QLAYERS, ROOT, FixedPointViT, embed, find_input, fixed_weights, float_forward,
    check_scales_source, float_weights_source, load_float_weights, load_mnist_test, load_scales,
    quantize_layers, read_head_mask, show, to_fixed,
)

WEIGHT_DIR = ROOT / "weights"
DATA_DIR = ROOT / "data"
MASK_DIR = ROOT / "head_masks"
QUANT_DIR = ROOT / "quant"
# Layer in QLAYERS -> file prefix in quant/blockN/.
QUANT_FILE = {"attention.wq": "wq", "attention.wk": "wk", "attention.wv": "wv", "fc1": "fc1", "fc2": "fc2"}

# File name in weights/blockN/ -> parameter name inside encoder.<N-1>.
BLOCK_TENSORS = {
    "norm1_weight": "norm1.weight",
    "norm1_bias": "norm1.bias",
    "wq_weight": "attention.wq.weight",
    "wq_bias": "attention.wq.bias",
    "wk_weight": "attention.wk.weight",
    "wk_bias": "attention.wk.bias",
    "wv_weight": "attention.wv.weight",
    "wv_bias": "attention.wv.bias",
    "norm2_weight": "norm2.weight",
    "norm2_bias": "norm2.bias",
    "fc1_weight": "fc1.weight",
    "fc1_bias": "fc1.bias",
    "fc2_weight": "fc2.weight",
    "fc2_bias": "fc2.bias",
}

OTHER_TENSORS = {
    "encoder_norm/norm_weight": "encoder.6.weight",
    "encoder_norm/norm_bias": "encoder.6.bias",
    "model_norm/norm_weight": "norm.weight",
    "model_norm/norm_bias": "norm.bias",
    "classifier/fc1_weight": "classifier.fc1.weight",
    "classifier/fc1_bias": "classifier.fc1.bias",
    "classifier/fc2_weight": "classifier.fc2.weight",
    "classifier/fc2_bias": "classifier.fc2.bias",
}


def write_column(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(v)) for v in np.ravel(values)) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(" ".join(str(int(v)) for v in np.ravel(r)) for r in rows) + "\n")


def export_weights(wq):
    for b in range(BLOCKS):
        for name, key in BLOCK_TENSORS.items():
            write_column(WEIGHT_DIR / f"block{b + 1}" / f"{name}.txt", wq[f"encoder.{b}.{key}"])
    for name, key in OTHER_TENSORS.items():
        write_column(WEIGHT_DIR / f"{name}.txt", wq[key])


def export_quant(quant):
    for b in range(BLOCKS):
        for layer in QLAYERS:
            q = quant[f"encoder.{b}.{layer}"]
            base = QUANT_DIR / f"block{b + 1}" / QUANT_FILE[layer]
            write_column(base.with_name(base.name + "_w8.txt"), q["w8"])
            write_column(base.with_name(base.name + "_x_mult.txt"), [q["x_mult"]])
            write_column(base.with_name(base.name + "_y_mult.txt"), q["y_mult"])


def export_golden(model, x_fixed, x_float, w, masks, out_dir, labels, tag):
    for case, mask in masks.items():
        results = [model.forward(x, mask) for x in x_fixed]
        golden_logits = np.array([r[0] for r in results])
        golden_pred = np.array([r[1] for r in results])
        float_pred = float_forward(x_float, w, mask).argmax(axis=1)
        write_rows(out_dir / case / "golden_logits.txt", golden_logits)
        write_column(out_dir / case / "golden_pred.txt", golden_pred)
        print(f"[{tag}{case}] samples={len(labels)} "
              f"fixed_acc={np.mean(golden_pred == labels):.4f} "
              f"float_acc={np.mean(float_pred == labels):.4f} "
              f"fixed_vs_float_agree={np.mean(golden_pred == float_pred):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Export Q4.12 weights, test inputs and golden outputs for the RTL testbench.")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--cases", nargs="*", default=None, help="head mask names under head_masks/ (default: all)")
    parser.add_argument("--int8", type=str, nargs="?", const="scales_ptq.npz", default=None, metavar="SCALES",
                        help="also export the LINEAR_INT8 parameters and golden data; SCALES is a name in "
                             "quant/ or a path (default scales_ptq.npz)")
    args = parser.parse_args()
    if not 1 <= args.samples <= 10000:
        raise SystemExit("--samples must be between 1 and 10000 (the MNIST test set)")

    # Check every input before anything is written.
    cases = args.cases or sorted(p.stem for p in MASK_DIR.glob("*.txt"))
    if "int8" in cases:
        raise SystemExit("rename head_masks/int8.txt: the name int8 is used by the INT8 golden data (data/int8/)")
    masks = {}
    for case in cases:
        if not (MASK_DIR / f"{case}.txt").is_file():
            raise SystemExit(f"no head mask head_masks/{case}.txt")
        masks[case] = read_head_mask(MASK_DIR / f"{case}.txt")
    scales_file = find_input(args.int8, "quant", ".npz") if args.int8 else None
    scales = load_scales(scales_file) if scales_file else None
    if scales is not None:
        check_scales_source(scales, scales_file)

    # The notes are written last: until then the exported data count as incomplete.
    int8_dir = DATA_DIR / "int8"
    had_int8 = int8_dir.exists()
    (DATA_DIR / "source.txt").unlink(missing_ok=True)
    if had_int8:
        shutil.rmtree(int8_dir)

    source = float_weights_source()
    w = load_float_weights()
    wq = fixed_weights(w)
    export_weights(wq)

    images, labels = load_mnist_test(args.samples)
    x_float = embed(images, w)
    x_fixed = to_fixed(x_float)
    write_rows(DATA_DIR / "embedding_input.txt", x_fixed.reshape(args.samples, -1))
    write_column(DATA_DIR / "label.txt", labels)
    (DATA_DIR / "sample_count.txt").write_text(f"{args.samples}\n")

    export_golden(FixedPointViT(wq), x_fixed, x_float, w, masks, DATA_DIR, labels, "")
    print(f"Weights written to {WEIGHT_DIR}")
    print(f"Test data written to {DATA_DIR}")

    if scales is not None:
        quant = quantize_layers(w, scales)
        export_quant(quant)
        export_golden(FixedPointViT(wq, quant), x_fixed, x_float, w, masks, int8_dir, labels, "int8 ")
        (int8_dir / "source.txt").write_text(f"weights of {source}, INT8 scales {show(scales_file)}\n")
        print(f"INT8 parameters written to {QUANT_DIR}")
        print(f"INT8 test data written to {int8_dir}")
    elif had_int8:
        print("Removed the INT8 test data (data/int8/): the INT8 build also uses weights/, which "
              "this export replaced. Export with --int8 before simulating the INT8 build again.")
    (DATA_DIR / "source.txt").write_text(f"weights of {source}\n")


if __name__ == "__main__":
    main()
