"""Accuracy of the hardware models and the float model on the MNIST test set for one head mask.

Without --int8 it compares the float model with the Q4.12 model of the 16-bit build; with
--int8 [SCALES] it also evaluates the LINEAR_INT8 build with those scales (a name in quant/
or a path; default quant/scales_ptq.npz).
"""

import argparse

import numpy as np

from vit_fixed import (
    FixedPointViT, check_scales_source, embed, find_input, fixed_weights, float_forward,
    float_weights_source, load_float_weights, load_mnist_test, load_scales, quantize_layers,
    read_head_mask, show, to_fixed,
)


def main():
    parser = argparse.ArgumentParser(description="Compare the fixed-point hardware models with the floating-point model on the MNIST test set.")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--case", type=str, default="keep_all", help="head mask: a name in head_masks/ or a path")
    parser.add_argument("--int8", type=str, nargs="?", const="scales_ptq.npz", default=None, metavar="SCALES",
                        help="also evaluate the LINEAR_INT8 build; SCALES is a name in quant/ or a path (default scales_ptq.npz)")
    args = parser.parse_args()

    source = float_weights_source()
    w = load_float_weights()
    wq = fixed_weights(w)
    models = {"Q4.12": FixedPointViT(wq)}
    if args.int8:
        scales_file = find_input(args.int8, "quant", ".npz")
        scales = load_scales(scales_file)
        check_scales_source(scales, scales_file)
        models["INT8"] = FixedPointViT(wq, quantize_layers(w, scales))
        source += f", INT8 scales {show(scales_file)}"
    mask_file = find_input(args.case, "head_masks", ".txt")
    mask = read_head_mask(mask_file)
    images, labels = load_mnist_test(args.samples)

    float_pred = []
    pred = {name: [] for name in models}
    for start in range(0, args.samples, 500):
        x = embed(images[start:start + 500], w)
        float_pred.append(float_forward(x, w, mask).argmax(axis=1))
        for name, model in models.items():
            pred[name].append([model.forward(xi, mask)[1] for xi in to_fixed(x)])
    float_pred = np.concatenate(float_pred)

    print(f"case={mask_file.stem} samples={args.samples}")
    print(f"weights of {source}")
    print(f"float accuracy : {np.mean(float_pred == labels):.4%}")
    for name in models:
        p = np.concatenate(pred[name])
        print(f"{name:5s} accuracy : {np.mean(p == labels):.4%}   (agrees with float on {np.mean(p == float_pred):.4%})")


if __name__ == "__main__":
    main()
