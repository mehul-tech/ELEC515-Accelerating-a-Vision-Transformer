"""Post-training quantization (PTQ) of the linear layers to INT8.

Quantized layers: the Q, K and V projections, FC1 and FC2 of each of the 6 encoder blocks
(30 layers). Everything else -- LayerNorm, softmax, GELU, the Q*K^T and P*V products and the
classifier -- stays 16-bit fixed point and is not touched.

Quantization is symmetric with zero point 0 and 8-bit signed integers in [-127, 127]:
    q = clip(round(r / s), -127, 127),   r ~ s * q
Each layer needs two kinds of scale:
    weights      one scale per output channel (row of W):   s_w = max|W_row| / 127
                 or, with --per_tensor_weights, one scale for the whole matrix
    activations  one scale per layer (per tensor):           s_x = max|x| / 127
                 where max|x| is the largest input magnitude the layer sees while the model
                 runs on a few calibration batches of *training* images
Calibration runs with every head kept, so one set of scales serves every head mask.
The scales are saved to quant/scales_ptq.npz. export_rtl_data.py --int8 then turns them
into the integers the LINEAR_INT8 hardware uses (int8 weights and integer multipliers):
    python quantize_ptq.py
    python export_rtl_data.py --samples 20 --int8
"""

import argparse

import numpy as np
import torch

from data import evaluate, get_loaders
from model import build_mnist_vit
from vit_fixed import QLAYERS, QMAX, checkpoint_id, find_input, output_path, show


# Returns the weight scales of one layer as a numpy array with one value per output channel.
def weight_scale(weight, per_channel):
    w = weight.detach().abs()          # shape [out_features][in_features]
    # TODO (1): compute the symmetric INT8 weight scales
    # HINT: per channel, the largest |w| of each row (output channel) must map to QMAX = 127;
    # HINT: per tensor, use the largest |w| of the whole matrix and repeat it for every row.
    # HINT: return a numpy array of length out_features (w.shape[0]); avoid zero scales
    # HINT: (clamp_min). Two lines: the per-row maxima, then the scale.
    raise NotImplementedError("TODO (1): compute the symmetric INT8 weight scales")


# Calibration observer, attached to a layer with register_forward_pre_hook: it is called with
# the layer's input before every forward pass and keeps the largest |x| seen so far.
class InputRange:
    def __init__(self):
        self.max_abs = 0.0

    def __call__(self, module, inputs):
        x = inputs[0].detach()
        # TODO (2): update the running maximum of the input magnitude
        # HINT: set self.max_abs to the larger of self.max_abs and max|x| of this batch, as a
        # HINT: Python float (.item()). One line.
        raise NotImplementedError("TODO (2): update the running maximum of the input magnitude")


# The 30 quantized layers, as (name, nn.Linear) pairs; names match vit_fixed.QLAYERS.
def quantized_layers(model):
    for b, block in enumerate(model.encoder[:-1]):
        for layer in QLAYERS:
            module = block
            for part in layer.split("."):
                module = getattr(module, part)
            yield f"encoder.{b}.{layer}", module


def main():
    parser = argparse.ArgumentParser(description="Post-training quantization of the linear layers to INT8.")
    parser.add_argument("--checkpoint", type=str, default="ViT_model.pt", help="a name in checkpoint/ or a path")
    parser.add_argument("--out", type=str, default="scales_ptq.npz", help="output: a file name (saved in quant/) or a path")
    parser.add_argument("--calib_batches", type=int, default=8, help="training batches used for calibration")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--per_tensor_weights", action="store_true", help="one weight scale per layer")
    args = parser.parse_args()

    torch.manual_seed(0)
    checkpoint = find_input(args.checkpoint, "checkpoint", ".pt")
    model = build_mnist_vit()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    train_loader, test_loader = get_loaders(args.batch_size, 0)

    # Calibration: run a few training batches with an observer on every quantized layer.
    observers, hooks = {}, []
    for name, module in quantized_layers(model):
        observers[name] = InputRange()
        hooks.append(module.register_forward_pre_hook(observers[name]))
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i >= args.calib_batches:
                break
            model(x)
    for h in hooks:
        h.remove()

    scales = {}
    for name, module in quantized_layers(model):
        # TODO (3): the activation scale of this layer
        # HINT: symmetric INT8, so the largest input magnitude seen during calibration
        # HINT: (observers[name].max_abs) maps to QMAX = 127. Store the scale in
        # HINT: scales[name + ".x_scale"] as a 0-d numpy array (np.array(value)). One line.
        raise NotImplementedError("TODO (3): the activation scale of this layer")
        scales[name + ".w_scale"] = weight_scale(module.weight, not args.per_tensor_weights)
    out = output_path(args.out, "quant", ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    # The scales fit only this checkpoint; export_rtl_data.py checks it against weights_float/.
    np.savez(out, checkpoint=np.array(checkpoint_id(checkpoint)), **scales)
    print(f"float test accuracy of {show(checkpoint)} (every head kept): "
          f"{evaluate(model, test_loader, torch.device('cpu')):.2%}")
    print(f"scales of {len(scales) // 2} layers written to {out}")


if __name__ == "__main__":
    main()
