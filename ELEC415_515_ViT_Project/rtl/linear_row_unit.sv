`include "fixed_params.vh"

// One linear unit: one token row through a linear layer (combinational).
//   y[j] = sum over d of x[d] * W[row_base + j][d]  +  b[row_base + j],   j = 0 .. OUT_DIM-1
// W is the whole [W_ROWS][IN_DIM] weight matrix; row_base selects OUT_DIM consecutive output
// channels (the 16 channels of one head, or all channels with row_base = 0).
// A layer over all T tokens is computed by LIN_UNITS of these units, one token each per cycle.
//
// 16-bit build: Q4.12 operands, Q8.24 accumulator.
//   y_full = sum + (b << 12)                         Q8.24, 64 bit (Q and K)
//   y      = sat16(y_full >>> 12)                    Q4.12
// LINEAR_INT8 build: symmetric int8 operands, integer accumulator.
//   x_q    = clip((x * x_mult + 2^23) >>> 24, -127, 127)       Q4.12 input -> int8
//   acc    = sum over d of x_q[d] * w8[row_base + j][d]
//   y      = sat16(((acc * y_mult[row_base + j] + 2^23) >>> 24) + b)     back to Q4.12
//   y_full = y << 12                                 Q8.24 (Q and K)
// x_mult and y_mult are computed from the real-valued scales by python/vit_fixed.py
// (quantize_layers); the bias stays Q4.12 in both builds.
module linear_row_unit #(
    parameter int IN_DIM  = `D,
    parameter int OUT_DIM = `D,
    parameter int W_ROWS  = `D
) (
    input  logic signed [IN_DIM*`WIDTH-1:0]        x_row,
    input  logic [15:0]                            row_base,
    // 16-bit build
    input  logic signed [W_ROWS*IN_DIM*`WIDTH-1:0] w_flat,
    input  logic signed [W_ROWS*`WIDTH-1:0]        b_flat,
    // LINEAR_INT8 build
    input  logic signed [W_ROWS*IN_DIM*8-1:0]      w8_flat,
    input  logic [31:0]                            x_mult,
    input  logic [W_ROWS*32-1:0]                   y_mult_flat,

    output logic signed [OUT_DIM*64-1:0]           y_full_row,
    output logic signed [OUT_DIM*`WIDTH-1:0]       y_row
);

    `include "fixed_math.vh"

`ifdef LINEAR_INT8
    // Activation quantized once per input element, then shared by all output channels.
    logic signed [IN_DIM*8-1:0] x_q_row;

    always_comb begin
        logic signed [63:0] xv, xm, t;
        xm = $signed({32'd0, x_mult});
        for (int d = 0; d < IN_DIM; d = d + 1) begin
            xv = $signed(`ELEM(x_row, d));
            t = (xv * xm + (64'sd1 <<< (`QSHIFT - 1))) >>> `QSHIFT;
            if (t > 64'sd127)
                t = 64'sd127;
            else if (t < -64'sd127)
                t = -64'sd127;
            `ELEM_W(x_q_row, d, 8) = t[7:0];
        end
    end

    always_comb begin
        logic signed [63:0] acc, xq, wq, ym, bv, y, y16;
        y_full_row = '0;
        y_row = '0;
        for (int j = 0; j < OUT_DIM; j = j + 1) begin
            acc = 64'sd0;
            for (int d = 0; d < IN_DIM; d = d + 1) begin
                xq = $signed(`ELEM_W(x_q_row, d, 8));
                wq = $signed(`ELEM_W(w8_flat, (int'(row_base) + j)*IN_DIM + d, 8));
                acc = acc + xq * wq;
            end
            ym = $signed({32'd0, `ELEM_W(y_mult_flat, int'(row_base) + j, 32)});
            bv = $signed(`ELEM(b_flat, int'(row_base) + j));
            y = ((acc * ym + (64'sd1 <<< (`QSHIFT - 1))) >>> `QSHIFT) + bv;
            y16 = sat16(y);                     // sign-extended to 64 bit
            `ELEM(y_row, j) = y16[`WIDTH-1:0];
            `ELEM_W(y_full_row, j, 64) = y16 <<< `FRAC;
        end
    end
`else
    always_comb begin
        logic signed [63:0] acc, xv, wv, bv;
        y_full_row = '0;
        y_row = '0;
        for (int j = 0; j < OUT_DIM; j = j + 1) begin
            bv = $signed(`ELEM(b_flat, int'(row_base) + j));
            acc = bv <<< `FRAC;
            for (int d = 0; d < IN_DIM; d = d + 1) begin
                xv = $signed(`ELEM(x_row, d));
                wv = $signed(`ELEM(w_flat, (int'(row_base) + j)*IN_DIM + d));
                acc = acc + xv * wv;
            end
            `ELEM_W(y_full_row, j, 64) = acc;
            `ELEM(y_row, j) = sat16(acc >>> `FRAC);
        end
    end
`endif

endmodule
