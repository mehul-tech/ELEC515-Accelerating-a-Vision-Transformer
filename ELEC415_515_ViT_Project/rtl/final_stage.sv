`include "fixed_params.vh"

// Stages after the last encoder block. The unit holds one image at a time:
//   ENC   encoder LayerNorm
//   MOD   model LayerNorm
//   CLS   classifier on the CLS token (logits and pred_class)
//   DONE  hold the result until it is taken
// Cycles from accepting an image to presenting the result: 3.
// Handshake: an image is accepted when in_valid && in_ready (in_ready is high only in IDLE)
// and the result is passed on when out_valid && out_ready.
module final_stage (
    input  logic clk,
    input  logic rst,

    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic [15:0]                  in_image_id,
    input  logic signed [`MAT_T_D_W-1:0] x_flat,

    input  signed [`VEC_D_W-1:0] encoder_norm_w_flat,
    input  signed [`VEC_D_W-1:0] encoder_norm_b_flat,
    input  signed [`VEC_D_W-1:0] model_norm_w_flat,
    input  signed [`VEC_D_W-1:0] model_norm_b_flat,
    input  signed [`W_D_D_W-1:0] classifier_fc1_w_flat,
    input  signed [`VEC_D_W-1:0] classifier_fc1_b_flat,
    input  signed [`W_C_D_W-1:0] classifier_fc2_w_flat,
    input  signed [`VEC_C_W-1:0] classifier_fc2_b_flat,

    output logic                           out_valid,
    input  logic                           out_ready,
    output logic [15:0]                    out_image_id,
    output logic signed [`VEC_LOGIT_W-1:0] logits_flat,
    output logic [3:0]                     pred_class,

    // High while an image is in the unit.
    output logic                           busy
);

    typedef enum logic [2:0] {S_IDLE, S_ENC, S_MOD, S_CLS, S_DONE} state_t;

    state_t state;
    logic [15:0] image_id_r;
    logic signed [`MAT_T_D_W-1:0] x_r;
    logic signed [`MAT_T_D_W-1:0] encoder_norm_r;
    logic signed [`MAT_T_D_W-1:0] pre_classifier_r;
    logic signed [`MAT_T_D_W-1:0] encoder_norm_comb;
    logic signed [`MAT_T_D_W-1:0] pre_classifier_comb;
    logic signed [`VEC_LOGIT_W-1:0] logits_comb;
    logic [3:0] pred_class_comb;

    assign in_ready     = (state == S_IDLE);
    assign out_valid    = (state == S_DONE);
    assign out_image_id = image_id_r;
    assign busy         = (state != S_IDLE);

    norm_stage_50x64 u_encoder_norm (
        .x_flat(x_r),
        .w_flat(encoder_norm_w_flat),
        .b_flat(encoder_norm_b_flat),
        .y_flat(encoder_norm_comb)
    );

    norm_stage_50x64 u_model_norm (
        .x_flat(encoder_norm_r),
        .w_flat(model_norm_w_flat),
        .b_flat(model_norm_b_flat),
        .y_flat(pre_classifier_comb)
    );

    classifier_head_64x10 u_classifier (
        .pre_classifier_flat(pre_classifier_r),
        .fc1_w_flat(classifier_fc1_w_flat),
        .fc1_b_flat(classifier_fc1_b_flat),
        .fc2_w_flat(classifier_fc2_w_flat),
        .fc2_b_flat(classifier_fc2_b_flat),
        .logits_flat(logits_comb),
        .pred_class(pred_class_comb)
    );

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            image_id_r <= 16'd0;
            x_r <= '0;
            encoder_norm_r <= '0;
            pre_classifier_r <= '0;
            logits_flat <= '0;
            pred_class <= 4'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (in_valid) begin
                        x_r <= x_flat;
                        image_id_r <= in_image_id;
                        state <= S_ENC;
                    end
                end
                S_ENC: begin
                    encoder_norm_r <= encoder_norm_comb;
                    state <= S_MOD;
                end
                S_MOD: begin
                    pre_classifier_r <= pre_classifier_comb;
                    state <= S_CLS;
                end
                S_CLS: begin
                    logits_flat <= logits_comb;
                    pred_class <= pred_class_comb;
                    state <= S_DONE;
                end
                S_DONE: begin
                    if (out_ready)
                        state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
