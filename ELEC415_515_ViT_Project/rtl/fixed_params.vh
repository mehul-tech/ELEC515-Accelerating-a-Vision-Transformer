`ifndef VIT_FIXED_PARAMS_VH
`define VIT_FIXED_PARAMS_VH

// Fixed-point format: signed 16-bit Q4.12 (4 integer bits including sign, 12 fractional bits).
// Range [-8, 8), resolution 2^-12. Attention scores and logits use 32-bit words with the
// same 12 fractional bits.
`define WIDTH 16
`define FRAC  12
`define SCORE_WIDTH 32
`define LOGIT_WIDTH 32

// Model dimensions: T tokens (49 patches + CLS), D embedding size, HEADS x HEAD_DIM = D,
// DFF hidden size of the MLP, BLOCKS encoder blocks.
`define T 50
`define D 64
`define HEADS 4
`define HEAD_DIM 16
`define DFF 128
`define N_CLASSES 10
`define BLOCKS 6

// Widths of flattened buses. A matrix is stored row-major in one vector: element (i, j)
// of a T x D matrix is at index i*D + j, element 0 in the least significant bits.
// Weight matrices use the PyTorch nn.Linear layout [out][in].
`define MAT_T_D_W     (`T*`D*`WIDTH)
`define MAT_T_DFF_W   (`T*`DFF*`WIDTH)
`define VEC_D_W       (`D*`WIDTH)
`define VEC_DFF_W     (`DFF*`WIDTH)
`define VEC_C_W       (`N_CLASSES*`WIDTH)
`define VEC_LOGIT_W   (`N_CLASSES*`LOGIT_WIDTH)
`define W_D_D_W       (`D*`D*`WIDTH)
`define W_DFF_D_W     (`DFF*`D*`WIDTH)
`define W_D_DFF_W     (`D*`DFF*`WIDTH)
`define W_C_D_W       (`N_CLASSES*`D*`WIDTH)

// INT8 linear layers (LINEAR_INT8 build): int8 weights and 32-bit integer multipliers.
`define W8_D_D_W      (`D*`D*8)
`define W8_DFF_D_W    (`DFF*`D*8)
`define W8_D_DFF_W    (`D*`DFF*8)
`define MULT_D_W      (`D*32)
`define MULT_DFF_W    (`DFF*32)
`define QSHIFT        24

// Linear units per linear layer: each unit processes one token per cycle, so a linear layer
// over the T tokens takes ceil(T / LIN_UNITS) cycles. An 8-bit multiplier needs about a
// quarter of the area of a 16-bit one, so four int8 multiplier arrays take about the area
// of one 16-bit array (the int8 units also need small per-channel multipliers to rescale
// their inputs and outputs, not counted here). Compile with +define+LINEAR_INT8 to select it.
`ifdef LINEAR_INT8
`define LIN_UNITS 4
`else
`define LIN_UNITS 1
`endif

// LayerNorm epsilon in LSB (1 = 2^-12).
`define EPS 1

// ELEM(bus, idx): 16-bit element idx of a flat bus. ELEM_W: the same with element width w.
`define ELEM(bus, idx) bus[((idx)+1)*`WIDTH-1 -: `WIDTH]
`define ELEM_W(bus, idx, w) bus[((idx)+1)*(w)-1 -: (w)]

`endif
