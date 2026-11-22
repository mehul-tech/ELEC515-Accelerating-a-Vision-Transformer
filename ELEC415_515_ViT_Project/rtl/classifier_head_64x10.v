`include "fixed_params.vh"

// Classifier on the CLS token, i.e. row 0 of the input (combinational):
//   h          = tanh(fc1(x0))   64 -> 64, tanh by lookup table
//   logits     = fc2(h)          64 -> 10, saturated to 32 bit with 12 fractional bits
//   pred_class = index of the largest logit (the lowest index on a tie)
module classifier_head_64x10 (
    input  signed [`MAT_T_D_W-1:0]  pre_classifier_flat,
    input  signed [`W_D_D_W-1:0]    fc1_w_flat,
    input  signed [`VEC_D_W-1:0]    fc1_b_flat,
    input  signed [`W_C_D_W-1:0]    fc2_w_flat,
    input  signed [`VEC_C_W-1:0]    fc2_b_flat,
    output reg signed [`VEC_LOGIT_W-1:0] logits_flat,
    output reg [3:0] pred_class
);

    `include "fixed_math.vh"
    `include "tanh_lut.vh"

    integer i, j, c;
    reg signed [63:0] acc, xv, wv, bv;
    reg signed [`WIDTH-1:0] tanh_y [0:`D-1];
    reg signed [`LOGIT_WIDTH-1:0] logits [0:`N_CLASSES-1];
    reg signed [`LOGIT_WIDTH-1:0] best_logit;

    always @(*) begin
        logits_flat = {`VEC_LOGIT_W{1'b0}};
        pred_class = 4'd0;

        for (i = 0; i < `D; i = i + 1) begin
            acc = 64'sd0;
            for (j = 0; j < `D; j = j + 1) begin
                xv = $signed(`ELEM(pre_classifier_flat, j));
                wv = $signed(`ELEM(fc1_w_flat, i*`D+j));
                acc = acc + xv * wv;
            end
            bv = $signed(`ELEM(fc1_b_flat, i));
            acc = acc + (bv <<< `FRAC);
            tanh_y[i] = tanh_lut_lookup(sat16(acc >>> `FRAC));
        end

        for (c = 0; c < `N_CLASSES; c = c + 1) begin
            acc = 64'sd0;
            for (j = 0; j < `D; j = j + 1) begin
                xv = tanh_y[j];
                wv = $signed(`ELEM(fc2_w_flat, c*`D+j));
                acc = acc + xv * wv;
            end
            bv = $signed(`ELEM(fc2_b_flat, c));
            acc = acc + (bv <<< `FRAC);
            logits[c] = sat32(acc >>> `FRAC);
            `ELEM_W(logits_flat, c, `LOGIT_WIDTH) = logits[c];
        end

        best_logit = logits[0];
        for (c = 1; c < `N_CLASSES; c = c + 1) begin
            if (logits[c] > best_logit) begin
                best_logit = logits[c];
                pred_class = c[3:0];
            end
        end
    end

endmodule
