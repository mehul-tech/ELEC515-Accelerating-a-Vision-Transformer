"""Bit-exact Python model of the Q4.12 RTL, and the floating-point reference model.

Every integer step (shifts, saturation, truncating division, lookup tables) follows the
RTL, so FixedPointViT produces exactly the logits of the hardware. With INT8 parameters
(see quantize_layers) it models the LINEAR_INT8 build, in which the Q/K/V projections, FC1
and FC2 of every block use 8-bit operands.
"""

import gzip
import hashlib
import math
import os
from pathlib import Path

import numpy as np

# The project folder is located from this file, so every script works from any directory.
ROOT = Path(__file__).resolve().parents[1]
FLOAT_DIR = ROOT / "weights_float"
MNIST_RAW = ROOT / "data" / "mnist" / "MNIST" / "raw"


# A file given on the command line. A path to an existing file (absolute, or relative to the
# current directory) is used as it is; anything else is taken as a file name inside the
# project folder `folder`. Either way `suffix` may be left out. So `prune4`, `prune4.txt`,
# `prune4.v2` and `../head_masks/prune4.txt` all work from any directory.
def find_input(arg, folder, suffix):
    given = Path(arg).expanduser()
    inside = ROOT / folder / given.name
    candidates = [given, inside]
    if given.suffix != suffix:
        candidates = [given, given.with_name(given.name + suffix), inside, inside.with_name(inside.name + suffix)]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise SystemExit(f"cannot find '{arg}' (looked for {given.resolve()} and in {show(ROOT / folder)}/, "
                     f"with and without {suffix})")


# An output file given on the command line: a bare file name goes into the project folder
# `folder`; anything with a directory part (`./x.pt`, `../x.pt`, `/tmp/x.pt`) is used as it
# is, relative to the current directory. `suffix` is added when the name does not end in it.
def output_path(arg, folder, suffix):
    given = Path(arg).expanduser()
    if given.suffix != suffix:
        given = given.with_name(given.name + suffix)
    has_dir = os.sep in arg or "/" in arg
    return given.resolve() if has_dir else ROOT / folder / given.name


# A file as it is named in messages and notes: relative to the project folder when inside it.
def show(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# A checkpoint as it is named in notes: its file name and the start of its SHA-256, so that a
# checkpoint overwritten under the same name counts as a different one.
def checkpoint_id(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    return f"{show(path)} (sha256 {digest})"


# The checkpoint that weights_float/ was exported from (noted by export_float_weights.py).
def float_weights_source(folder=None):
    note = Path(folder or FLOAT_DIR) / "source.txt"
    return note.read_text().strip() if note.is_file() else "unknown"


# INT8 scales only fit the model they were calibrated on (noted by quantize_ptq.py); warn
# when that is not the model in weights_float/ (same name and same content).
def check_scales_source(scales, scales_file):
    calibrated = str(scales.get("checkpoint", "unknown"))
    exported = float_weights_source()
    if calibrated != exported:
        print(f"WARNING: {show(scales_file)} was calibrated on {calibrated}, but weights_float/ holds "
              f"{exported}. Run quantize_ptq.py --checkpoint with the exported checkpoint.")

# Model dimensions and fixed-point constants, as in rtl/fixed_params.vh.
T = 50
D = 64
HEADS = 4
HEAD_DIM = 16
DFF = 128
BLOCKS = 6
N_CLASSES = 10
FRAC = 12
ONE = 1 << FRAC
EPS = 1
MIN16, MAX16 = -(1 << 15), (1 << 15) - 1
MIN32, MAX32 = -(1 << 31), (1 << 31) - 1

# INT8 linear layers (LINEAR_INT8 build). Each listed layer of every block has a per-tensor
# activation scale and per-output-channel weight scales; QSHIFT is the right shift that
# follows both integer multipliers. Operands are symmetric int8 in [-127, 127].
QLAYERS = ("attention.wq", "attention.wk", "attention.wv", "fc1", "fc2")
QSHIFT = 24
QMAX = 127


def sat16(x):
    return np.clip(x, MIN16, MAX16).astype(np.int64)


def sat32(x):
    return np.clip(x, MIN32, MAX32).astype(np.int64)


# Float -> Q4.12: round to nearest, saturate to 16 bit.
def to_fixed(x):
    return sat16(np.floor(np.asarray(x, dtype=np.float64) * ONE + 0.5))


# Integer division rounding toward zero, like Verilog '/' on signed values.
def trunc_div(a, b):
    q = np.abs(a) // np.abs(b)
    return np.where((a < 0) != (b < 0), -q, q)


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


# Lookup tables in Q4.12; entry i belongs to the input (i - 32768) / 4096.
# exp covers [-8, 0]; GELU and tanh cover every 16-bit input.
def build_exp_lut():
    z = np.arange(MIN16, 1, dtype=np.int64)
    return sat16(np.floor(np.exp(z / ONE) * ONE + 0.5))


def build_gelu_lut():
    x = np.arange(MIN16, MAX16 + 1, dtype=np.int64)
    return sat16(np.floor(gelu(x / ONE) * ONE + 0.5))


def build_tanh_lut():
    x = np.arange(MIN16, MAX16 + 1, dtype=np.int64)
    return sat16(np.floor(np.tanh(x / ONE) * ONE + 0.5))


# Float weights from weights_float/, keyed by PyTorch parameter name.
def load_float_weights(folder=FLOAT_DIR):
    if not (Path(folder) / "source.txt").is_file():
        raise SystemExit(f"{show(folder)}/ has not been exported yet: run python/export_float_weights.py first")
    return {p.stem.replace("__", "."): np.load(p).astype(np.float64) for p in Path(folder).glob("*.npy")}


# Reads an MNIST IDX file, or its .gz version when the plain file is absent.
def _read_idx(path):
    path = Path(path)
    gz = path.with_name(path.name + ".gz")
    data = path.read_bytes() if path.exists() else gzip.open(gz, "rb").read()
    magic = int.from_bytes(data[0:4], "big")
    if magic == 2051:
        n, rows, cols = (int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(3))
        return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows, cols)
    return np.frombuffer(data, dtype=np.uint8, offset=8)


def load_mnist_test(count=None, raw_dir=MNIST_RAW):
    images = _read_idx(Path(raw_dir) / "t10k-images-idx3-ubyte")
    labels = _read_idx(Path(raw_dir) / "t10k-labels-idx1-ubyte")
    if count is not None:
        images, labels = images[:count], labels[:count]
    return images, labels.astype(np.int64)


# Patch embedding, CLS token and position embedding in float (computed outside the RTL).
# Returns (n, 50, 64): token 0 is CLS, tokens 1 .. 49 are the 4 x 4 patches in raster order.
def embed(images, w):
    x = (images.astype(np.float64) / 255.0 - 0.5) / 0.5
    n = x.shape[0]
    patches = x.reshape(n, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(n, 49, 16)
    conv_w = w["embedding.conv1.weight"].reshape(D, 16)
    tokens = patches @ conv_w.T + w["embedding.conv1.bias"]
    cls = np.broadcast_to(w["embedding.cls_token"].reshape(1, 1, D), (n, 1, D))
    return np.concatenate([cls, tokens], axis=1) + w["embedding.pos_embedding"].reshape(1, T, D)


def _layer_norm(x, weight, bias):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5) * weight + bias


# Floating-point forward pass from the embedded tokens to the logits, with an optional 6 x 4 head mask.
def float_forward(x, w, head_mask=None):
    mask = np.ones((BLOCKS, HEADS)) if head_mask is None else np.asarray(head_mask, dtype=np.float64)
    n = x.shape[0]
    for b in range(BLOCKS):
        p = f"encoder.{b}."
        h = _layer_norm(x, w[p + "norm1.weight"], w[p + "norm1.bias"])
        q = (h @ w[p + "attention.wq.weight"].T + w[p + "attention.wq.bias"]).reshape(n, T, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = (h @ w[p + "attention.wk.weight"].T + w[p + "attention.wk.bias"]).reshape(n, T, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = (h @ w[p + "attention.wv.weight"].T + w[p + "attention.wv.bias"]).reshape(n, T, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(HEAD_DIM)
        s = np.exp(s - s.max(axis=-1, keepdims=True))
        s = s / s.sum(axis=-1, keepdims=True)
        ctx = (s @ v) * mask[b].reshape(1, HEADS, 1, 1)
        x = x + ctx.transpose(0, 2, 1, 3).reshape(n, T, D)
        h = _layer_norm(x, w[p + "norm2.weight"], w[p + "norm2.bias"])
        h = gelu(h @ w[p + "fc1.weight"].T + w[p + "fc1.bias"])
        x = x + h @ w[p + "fc2.weight"].T + w[p + "fc2.bias"]
    x = _layer_norm(x, w["encoder.6.weight"], w["encoder.6.bias"])
    x = _layer_norm(x, w["norm.weight"], w["norm.bias"])
    c = np.tanh(x[:, 0, :] @ w["classifier.fc1.weight"].T + w["classifier.fc1.bias"])
    return c @ w["classifier.fc2.weight"].T + w["classifier.fc2.bias"]


# Quantize every tensor to Q4.12.
def fixed_weights(w):
    return {k: to_fixed(v) for k, v in w.items()}


def round_half_up(x):
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.int64)


# Real-valued scales -> the integers the LINEAR_INT8 hardware uses, for every layer in QLAYERS.
#   scales["encoder.<b>.<layer>.x_scale"]  activation scale s_x (one number)
#   scales["encoder.<b>.<layer>.w_scale"]  weight scales s_w, one per output channel
# For each layer the result holds
#   w8      int8 weights round(W / s_w), clipped to [-127, 127], shape [out][in]
#   x_mult  round(2^24 / (4096 * s_x)): a Q4.12 input x becomes x_q = (x * x_mult) >> 24
#   y_mult  round(s_x * s_w * 4096 * 2^24): an int32 sum acc becomes Q4.12 (acc * y_mult) >> 24
def quantize_layers(w, scales):
    quant = {}
    for b in range(BLOCKS):
        for layer in QLAYERS:
            name = f"encoder.{b}.{layer}"
            s_x = float(scales[name + ".x_scale"])
            s_w = np.broadcast_to(np.asarray(scales[name + ".w_scale"], dtype=np.float64),
                                  (w[name + ".weight"].shape[0],))
            if s_x <= 0 or np.any(s_w <= 0):
                raise ValueError(f"{name}: scales must be positive")
            x_mult = int(round_half_up((1 << QSHIFT) / (ONE * s_x)))
            y_mult = round_half_up(s_x * s_w * ONE * (1 << QSHIFT))
            if not (0 < x_mult < (1 << 31)) or np.any(y_mult <= 0) or np.any(y_mult >= (1 << 31)):
                raise ValueError(f"{name}: a multiplier does not fit in 31 bits; check the scales")
            w8 = np.clip(round_half_up(w[name + ".weight"] / s_w[:, None]), -QMAX, QMAX)
            quant[name] = {"w8": w8, "x_mult": x_mult, "y_mult": y_mult}
    return quant


# Scales saved by a quantization script (numpy .npz, keys as in quantize_layers, plus the
# name of the calibrated checkpoint under "checkpoint").
def load_scales(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# Integer model of the RTL; each method corresponds to an RTL module or stage.
# numpy '>>' on int64 is an arithmetic shift, the same as '>>>' in the RTL.
class FixedPointViT:
    # wq: Q4.12 weights (fixed_weights). quant: INT8 parameters (quantize_layers) for the
    # LINEAR_INT8 build, or None for the 16-bit build.
    def __init__(self, wq, quant=None):
        self.w = wq
        self.quant = quant
        self.exp_lut = build_exp_lut()
        self.gelu_lut = build_gelu_lut()
        self.tanh_lut = build_tanh_lut()

    # Accumulator before the final shift, Q8.24.
    def linear_acc(self, x, weight, bias):
        return x @ weight.T + (bias << FRAC)

    # Linear layer back to Q4.12 with saturation (fc1/fc2 stages, V projection).
    def linear(self, x, weight, bias):
        return sat16(self.linear_acc(x, weight, bias) >> FRAC)

    # INT8 linear layer of the LINEAR_INT8 build (rtl/linear_row_unit.sv), output Q4.12:
    #   x_q = clip((x * x_mult + 2^23) >> 24, -127, 127)     Q4.12 -> int8
    #   acc = x_q * w8^T                                     int32
    #   y   = sat16(((acc * y_mult + 2^23) >> 24) + bias)    back to Q4.12, Q4.12 bias
    def int8_linear(self, x, name, bias):
        q = self.quant[name]
        half = 1 << (QSHIFT - 1)
        x_q = np.clip((x * q["x_mult"] + half) >> QSHIFT, -QMAX, QMAX)
        acc = x_q @ q["w8"].T
        return sat16(((acc * q["y_mult"] + half) >> QSHIFT) + bias)

    # Same steps as rtl/norm_stage_50x64.v.
    def layer_norm(self, x, weight, bias):
        mean = x.sum(axis=-1, keepdims=True) >> 6
        dx = x - mean
        var = ((dx * dx).sum(axis=-1, keepdims=True) >> 6) >> FRAC
        std = np.vectorize(math.isqrt)((var + EPS) << FRAC).astype(np.int64)
        norm = trunc_div(dx << FRAC, std)
        return sat16(((norm * weight) >> FRAC) + bias)

    # States LN1 .. H6 of rtl/transformer_block_head_serial.sv. The hardware visits the kept
    # heads one at a time; here all heads are computed and the pruned ones are zeroed.
    def attention(self, x, p, mask_row):
        wq, wk, wv = self.w[p + "attention.wq.weight"], self.w[p + "attention.wk.weight"], self.w[p + "attention.wv.weight"]
        bq, bk, bv = self.w[p + "attention.wq.bias"], self.w[p + "attention.wk.bias"], self.w[p + "attention.wv.bias"]
        if self.quant is None:
            q_full = self.linear_acc(x, wq, bq)
            k_full = self.linear_acc(x, wk, bk)
            v = self.linear(x, wv, bv)
        else:
            # INT8 Q and K are Q4.12; << 12 puts them on the Q8.24 scale the score expects.
            q_full = self.int8_linear(x, p + "attention.wq", bq) << FRAC
            k_full = self.int8_linear(x, p + "attention.wk", bk) << FRAC
            v = self.int8_linear(x, p + "attention.wv", bv)
        q_full = q_full.reshape(T, HEADS, HEAD_DIM).transpose(1, 0, 2)
        k_full = k_full.reshape(T, HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = v.reshape(T, HEADS, HEAD_DIM).transpose(1, 0, 2)
        score = sat32(((q_full @ k_full.transpose(0, 2, 1)) >> 36) >> 2)
        centered = np.maximum(score - score.max(axis=-1, keepdims=True), MIN16)
        e = self.exp_lut[centered - MIN16]
        prob = (e << FRAC) // e.sum(axis=-1, keepdims=True)
        ctx = sat16((prob @ v) >> FRAC)
        ctx = ctx * np.asarray(mask_row, dtype=np.int64).reshape(HEADS, 1, 1)
        return ctx.transpose(1, 0, 2).reshape(T, D)

    # One encoder block: LayerNorm1 and attention, residual 1, then the MLP (RES1 .. RES2).
    def block(self, x, b, mask_row):
        p = f"encoder.{b}."
        h = self.layer_norm(x, self.w[p + "norm1.weight"], self.w[p + "norm1.bias"])
        x = sat16(x + self.attention(h, p, mask_row))
        h = self.layer_norm(x, self.w[p + "norm2.weight"], self.w[p + "norm2.bias"])
        if self.quant is None:
            h = self.gelu_lut[self.linear(h, self.w[p + "fc1.weight"], self.w[p + "fc1.bias"]) - MIN16]
            return sat16(x + self.linear(h, self.w[p + "fc2.weight"], self.w[p + "fc2.bias"]))
        h = self.gelu_lut[self.int8_linear(h, p + "fc1", self.w[p + "fc1.bias"]) - MIN16]
        return sat16(x + self.int8_linear(h, p + "fc2", self.w[p + "fc2.bias"]))

    # x: 50 x 64 Q4.12 tokens of one image. Returns the 32-bit logits and the predicted class.
    def forward(self, x, head_mask=None):
        mask = np.ones((BLOCKS, HEADS), dtype=np.int64) if head_mask is None else np.asarray(head_mask, dtype=np.int64)
        x = np.asarray(x, dtype=np.int64).reshape(T, D)
        for b in range(BLOCKS):
            x = self.block(x, b, mask[b])
        x = self.layer_norm(x, self.w["encoder.6.weight"], self.w["encoder.6.bias"])
        x = self.layer_norm(x, self.w["norm.weight"], self.w["norm.bias"])
        c = self.tanh_lut[self.linear(x[0], self.w["classifier.fc1.weight"], self.w["classifier.fc1.bias"]) - MIN16]
        logits = sat32(self.linear_acc(c, self.w["classifier.fc2.weight"], self.w["classifier.fc2.bias"]) >> FRAC)
        return logits, int(np.argmax(logits))


# Head mask file: 6 rows (blocks) x 4 columns (heads), 1 = keep.
def read_head_mask(path):
    values = Path(path).read_text().split()
    if len(values) != BLOCKS * HEADS or any(v not in ("0", "1") for v in values):
        raise SystemExit(f"{show(path)} is not a head mask: it must have {BLOCKS} rows of {HEADS} "
                         f"numbers, each 0 or 1 (only mask files belong in head_masks/)")
    return np.array([int(v) for v in values], dtype=np.int64).reshape(BLOCKS, HEADS)
