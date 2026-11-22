`include "fixed_params.vh"

// Element-wise saturating add of two T x D matrices (residual connection, combinational).
module residual_add_50x64 (
    input  signed [`MAT_T_D_W-1:0] a_flat,
    input  signed [`MAT_T_D_W-1:0] b_flat,
    output reg signed [`MAT_T_D_W-1:0] y_flat
);

    `include "fixed_math.vh"

    integer i;
    reg signed [63:0] av, bv;

    always @(*) begin
        y_flat = {`MAT_T_D_W{1'b0}};
        for (i = 0; i < `T*`D; i = i + 1) begin
            av = $signed(`ELEM(a_flat, i));
            bv = $signed(`ELEM(b_flat, i));
            `ELEM(y_flat, i) = sat16(av + bv);
        end
    end

endmodule
