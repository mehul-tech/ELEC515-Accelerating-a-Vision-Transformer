`include "fixed_params.vh"

// ViT datapath after the embedding: 6 encoder blocks, then the final LayerNorms and the
// classifier.
//   x_flat      50 x 64 Q4.12 token matrix of one image (patch embedding, CLS token and
//               position embedding are computed outside)
//   logits_flat 10 logits, 32 bit each; pred_class is the argmax
//   image_id    travels with the data and identifies each result
// The blocks and the final stage are chained with valid/ready handshakes; each holds one
// image, so up to 7 images are in flight. Every block streams the tokens through its
// compute units and computes attention one head at a time, skipping the heads that its
// row of the head mask prunes (see transformer_block_head_serial.sv).
// Build options (fixed_params.vh): the 16-bit build has one linear unit per linear layer;
// +define+LINEAR_INT8 quantizes the Q/K/V projections, FC1 and FC2 to int8 and has four.
// Cycles per block and image (L = 50 in the 16-bit build, 13 in the INT8 build):
//   occupancy  7 + 2L + kept_heads x (103 + L)
// Latency of an image alone in the hardware: sum over the 6 blocks of (occupancy - 1) + 3.
// Throughput: one image per (largest block occupancy) cycles, set by the slowest block.
// WEIGHT_ROOT: folder with the Q4.12 weight files.
// QUANT_ROOT: folder with the INT8 parameters (LINEAR_INT8 build only).
// head_mask: configuration input, one 4-bit mask per block (bit h = 1 keeps head h). Each
// block samples its mask when it accepts an image.
module vit_top #(
    parameter string WEIGHT_ROOT = "./weights",
    parameter string QUANT_ROOT = "./quant"
) (
    input  logic clk,
    input  logic rst,

    input  logic [`HEADS-1:0]            head_mask [0:`BLOCKS-1],

    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic [15:0]                  in_image_id,
    input  logic signed [`MAT_T_D_W-1:0] x_flat,

    output logic                         out_valid,
    input  logic                         out_ready,
    output logic [15:0]                  out_image_id,
    output logic signed [`VEC_LOGIT_W-1:0] logits_flat,
    output logic [3:0]                   pred_class,

    output logic [6:0]                   stage_busy
);

    // Handshake between neighbouring stages: block_*[k] is the input of block k
    // (k = 0 .. 5); block_*[6] is the input of the final stage.
    logic                         block_valid [0:`BLOCKS];
    logic                         block_ready [0:`BLOCKS];
    logic [15:0]                  block_image_id [0:`BLOCKS];
    logic signed [`MAT_T_D_W-1:0] block_data [0:`BLOCKS];
    logic                         stage_busy_bits [0:`BLOCKS];


    logic signed [`VEC_D_W-1:0]   n1w_flat     [0:`BLOCKS-1];
    logic signed [`VEC_D_W-1:0]   n1b_flat     [0:`BLOCKS-1];
    logic signed [`W_D_D_W-1:0]   wq_flat      [0:`BLOCKS-1];
    logic signed [`VEC_D_W-1:0]   bq_flat      [0:`BLOCKS-1];
    logic signed [`W_D_D_W-1:0]   wk_flat      [0:`BLOCKS-1];
    logic signed [`VEC_D_W-1:0]   bk_flat      [0:`BLOCKS-1];
    logic signed [`W_D_D_W-1:0]   wv_flat      [0:`BLOCKS-1];
    logic signed [`VEC_D_W-1:0]   bv_flat      [0:`BLOCKS-1];

    logic signed [`VEC_D_W-1:0]   n2w_flat     [0:`BLOCKS-1];
    logic signed [`VEC_D_W-1:0]   n2b_flat     [0:`BLOCKS-1];
    logic signed [`W_DFF_D_W-1:0] fc1_w_flat   [0:`BLOCKS-1];
    logic signed [`VEC_DFF_W-1:0] fc1_b_flat   [0:`BLOCKS-1];
    logic signed [`W_D_DFF_W-1:0] fc2_w_flat   [0:`BLOCKS-1];
    logic signed [`VEC_D_W-1:0]   fc2_b_flat   [0:`BLOCKS-1];

    logic signed [`W8_D_D_W-1:0]   wq8_flat     [0:`BLOCKS-1];
    logic signed [`W8_D_D_W-1:0]   wk8_flat     [0:`BLOCKS-1];
    logic signed [`W8_D_D_W-1:0]   wv8_flat     [0:`BLOCKS-1];
    logic signed [`W8_DFF_D_W-1:0] fc1_w8_flat  [0:`BLOCKS-1];
    logic signed [`W8_D_DFF_W-1:0] fc2_w8_flat  [0:`BLOCKS-1];
    logic [31:0]                   xm_q [0:`BLOCKS-1], xm_k [0:`BLOCKS-1], xm_v [0:`BLOCKS-1];
    logic [31:0]                   xm_fc1 [0:`BLOCKS-1], xm_fc2 [0:`BLOCKS-1];
    logic [`MULT_D_W-1:0]          ym_q_flat [0:`BLOCKS-1], ym_k_flat [0:`BLOCKS-1], ym_v_flat [0:`BLOCKS-1];
    logic [`MULT_DFF_W-1:0]        ym_fc1_flat [0:`BLOCKS-1];
    logic [`MULT_D_W-1:0]          ym_fc2_flat [0:`BLOCKS-1];

    logic signed [`VEC_D_W-1:0]   encoder_norm_w_flat;
    logic signed [`VEC_D_W-1:0]   encoder_norm_b_flat;
    logic signed [`VEC_D_W-1:0]   model_norm_w_flat;
    logic signed [`VEC_D_W-1:0]   model_norm_b_flat;

    logic signed [`W_D_D_W-1:0]   classifier_fc1_w_flat;
    logic signed [`VEC_D_W-1:0]   classifier_fc1_b_flat;
    logic signed [`W_C_D_W-1:0]   classifier_fc2_w_flat;
    logic signed [`VEC_C_W-1:0]   classifier_fc2_b_flat;

    vit_weight_memory #(
        .WEIGHT_ROOT(WEIGHT_ROOT)
    ) u_weight_memory (
        .n1w_flat(n1w_flat),
        .n1b_flat(n1b_flat),
        .wq_flat(wq_flat),
        .bq_flat(bq_flat),
        .wk_flat(wk_flat),
        .bk_flat(bk_flat),
        .wv_flat(wv_flat),
        .bv_flat(bv_flat),
        .n2w_flat(n2w_flat),
        .n2b_flat(n2b_flat),
        .fc1_w_flat(fc1_w_flat),
        .fc1_b_flat(fc1_b_flat),
        .fc2_w_flat(fc2_w_flat),
        .fc2_b_flat(fc2_b_flat),
        .encoder_norm_w_flat(encoder_norm_w_flat),
        .encoder_norm_b_flat(encoder_norm_b_flat),
        .model_norm_w_flat(model_norm_w_flat),
        .model_norm_b_flat(model_norm_b_flat),
        .classifier_fc1_w_flat(classifier_fc1_w_flat),
        .classifier_fc1_b_flat(classifier_fc1_b_flat),
        .classifier_fc2_w_flat(classifier_fc2_w_flat),
        .classifier_fc2_b_flat(classifier_fc2_b_flat)
    );

    vit_quant_memory #(
        .QUANT_ROOT(QUANT_ROOT)
    ) u_quant_memory (
        .wq8_flat(wq8_flat), .wk8_flat(wk8_flat), .wv8_flat(wv8_flat),
        .fc1_w8_flat(fc1_w8_flat), .fc2_w8_flat(fc2_w8_flat),
        .xm_q(xm_q), .xm_k(xm_k), .xm_v(xm_v), .xm_fc1(xm_fc1), .xm_fc2(xm_fc2),
        .ym_q_flat(ym_q_flat), .ym_k_flat(ym_k_flat), .ym_v_flat(ym_v_flat),
        .ym_fc1_flat(ym_fc1_flat), .ym_fc2_flat(ym_fc2_flat)
    );

    // An image enters when block 0 is free; a result leaves when the receiver takes it.
    // Backpressure travels from out_ready back to in_ready, so no result is ever lost.
    assign in_ready = block_ready[0];

    assign block_valid[0] = in_valid;
    assign block_image_id[0] = in_image_id;
    assign block_data[0] = x_flat;

    genvar bi;
    generate
        for (bi = 0; bi < `BLOCKS; bi = bi + 1) begin : g_blocks
            transformer_block_head_serial u_block (
                .clk(clk),
                .rst(rst),
                .in_valid(block_valid[bi]),
                .in_ready(block_ready[bi]),
                .in_image_id(block_image_id[bi]),
                .x_flat(block_data[bi]),
                .n1w_flat(n1w_flat[bi]),
                .n1b_flat(n1b_flat[bi]),
                .wq_flat(wq_flat[bi]),
                .bq_flat(bq_flat[bi]),
                .wk_flat(wk_flat[bi]),
                .bk_flat(bk_flat[bi]),
                .wv_flat(wv_flat[bi]),
                .bv_flat(bv_flat[bi]),
                .n2w_flat(n2w_flat[bi]),
                .n2b_flat(n2b_flat[bi]),
                .fc1_w_flat(fc1_w_flat[bi]),
                .fc1_b_flat(fc1_b_flat[bi]),
                .fc2_w_flat(fc2_w_flat[bi]),
                .fc2_b_flat(fc2_b_flat[bi]),
                .wq8_flat(wq8_flat[bi]),
                .wk8_flat(wk8_flat[bi]),
                .wv8_flat(wv8_flat[bi]),
                .fc1_w8_flat(fc1_w8_flat[bi]),
                .fc2_w8_flat(fc2_w8_flat[bi]),
                .xm_q(xm_q[bi]), .xm_k(xm_k[bi]), .xm_v(xm_v[bi]),
                .xm_fc1(xm_fc1[bi]), .xm_fc2(xm_fc2[bi]),
                .ym_q_flat(ym_q_flat[bi]), .ym_k_flat(ym_k_flat[bi]), .ym_v_flat(ym_v_flat[bi]),
                .ym_fc1_flat(ym_fc1_flat[bi]), .ym_fc2_flat(ym_fc2_flat[bi]),
                .head_mask(head_mask[bi]),
                .out_valid(block_valid[bi + 1]),
                .out_ready(block_ready[bi + 1]),
                .out_image_id(block_image_id[bi + 1]),
                .y_flat(block_data[bi + 1]),
                .busy(stage_busy_bits[bi])
            );
        end
    endgenerate

    final_stage u_final (
        .clk(clk),
        .rst(rst),
        .in_valid(block_valid[`BLOCKS]),
        .in_ready(block_ready[`BLOCKS]),
        .in_image_id(block_image_id[`BLOCKS]),
        .x_flat(block_data[`BLOCKS]),
        .encoder_norm_w_flat(encoder_norm_w_flat),
        .encoder_norm_b_flat(encoder_norm_b_flat),
        .model_norm_w_flat(model_norm_w_flat),
        .model_norm_b_flat(model_norm_b_flat),
        .classifier_fc1_w_flat(classifier_fc1_w_flat),
        .classifier_fc1_b_flat(classifier_fc1_b_flat),
        .classifier_fc2_w_flat(classifier_fc2_w_flat),
        .classifier_fc2_b_flat(classifier_fc2_b_flat),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_image_id(out_image_id),
        .logits_flat(logits_flat),
        .pred_class(pred_class),
        .busy(stage_busy_bits[`BLOCKS])
    );

    // stage_busy[k] = 1 while block k holds an image (bit 6: the final stage).
    generate
        for (bi = 0; bi <= `BLOCKS; bi = bi + 1) begin : g_busy
            assign stage_busy[bi] = stage_busy_bits[bi];
        end
    endgenerate

endmodule
