`include "fixed_params.vh"

// LayerNorm over the D channels of each of the T tokens (combinational).
//   mean = sum(x) >>> 6                          divide by D = 64
//   var  = (sum((x - mean)^2) >>> 6) >>> 12      Q4.12
//   std  = isqrt((var + EPS) << 12)              Q4.12
//   norm = ((x - mean) << 12) / std              Q4.12, division truncates toward zero
//   y    = sat16(((norm * w) >>> 12) + b)
// All intermediates are 64 bit.
module norm_stage_50x64 (
    input  signed [`MAT_T_D_W-1:0] x_flat,
    input  signed [`VEC_D_W-1:0]   w_flat,
    input  signed [`VEC_D_W-1:0]   b_flat,
    output reg signed [`MAT_T_D_W-1:0] y_flat
);

    `include "fixed_math.vh"

    integer i, j;
    reg signed [63:0] xv, wv, bv;
    reg signed [63:0] sum, sumsq, mean, dx;
    reg signed [63:0] var_fixed, std_fixed, norm;

    always @(*) begin
        y_flat = {`MAT_T_D_W{1'b0}};
        for (i = 0; i < `T; i = i + 1) begin
            sum = 64'sd0;
            for (j = 0; j < `D; j = j + 1) begin
                xv = $signed(`ELEM(x_flat, i*`D+j));
                sum = sum + xv;
            end
            mean = sum >>> 6;

            sumsq = 64'sd0;
            for (j = 0; j < `D; j = j + 1) begin
                xv = $signed(`ELEM(x_flat, i*`D+j));
                dx = xv - mean;
                sumsq = sumsq + dx * dx;
            end
            // sumsq is Q8.24: >>> 6 divides by D, >>> FRAC returns to Q4.12.
            var_fixed = (sumsq >>> 6) >>> `FRAC;
            // The square root of a Q8.24 value is Q4.12.
            std_fixed = $signed({32'd0, isqrt((var_fixed + `EPS) <<< `FRAC)});

            for (j = 0; j < `D; j = j + 1) begin
                xv = $signed(`ELEM(x_flat, i*`D+j));
                wv = $signed(`ELEM(w_flat, j));
                bv = $signed(`ELEM(b_flat, j));
                dx = xv - mean;
                norm = (dx <<< `FRAC) / std_fixed;
                `ELEM(y_flat, i*`D+j) = sat16(((norm * wv) >>> `FRAC) + bv);
            end
        end
    end

endmodule
