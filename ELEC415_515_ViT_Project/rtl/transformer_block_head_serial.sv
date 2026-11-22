`include "fixed_params.vh"

// One pre-norm encoder block with token-serial compute units and a head-serial attention
// engine:
//   x   = x + Attention(LayerNorm1(x))
//   out = x + FC2(GELU(FC1(LayerNorm2(x))))
//
// The block holds one image at a time and steps through a state machine. Every matrix
// multiplication streams the T = 50 tokens through a compute unit, one token row per unit
// per cycle; the cheap element-wise steps take a single cycle for the whole matrix:
//
//   state  computes                                                 cycles
//   IDLE   accept the image (x, head mask)                          1
//   LN1    LayerNorm1                                               1
//   H1     Q_h, K_h, V_h: the 16 channels of head h        |        ceil(50 / LIN_UNITS)
//   H2     S_h = Q_h * K_h^T / sqrt(16), one query row/cycle|        50
//   H3     subtract the row maximum                         |  per   1
//   H4     exp lookup and row sum                           |  kept  1
//   H5     divide -> probabilities P_h                      |  head  1
//   H6     context_h = P_h * V_h, one row per cycle,       |        50
//          written to context channels 16h .. 16h+15        |
//   RES1   residual 1: x + context                                  1
//   LN2    LayerNorm2                                               1
//   FC1    FC1 (64 -> 128)                                          ceil(50 / LIN_UNITS)
//   GELU   GELU                                                     1
//   FC2    FC2 (128 -> 64)                                          ceil(50 / LIN_UNITS)
//   RES2   residual 2                                               1
//   DONE   hold y until the next stage takes it                     1 (or more, while
//                                                                     the next stage is busy)
//
// Linear layers (Q/K/V, FC1, FC2) use LIN_UNITS linear units (linear_row_unit.sv): one in
// the 16-bit build, four in the LINEAR_INT8 build (see fixed_params.vh), so a linear layer
// takes 50 or 13 cycles. The Q/K/V units compute only the 16 channels of the current head,
// so H1 runs once per kept head. Q*K^T and P*V multiply activations by activations; they
// are not quantized and use one unit in both builds. In the 16-bit build Q and K keep their
// full-precision Q8.24 sums for Q*K^T; in the INT8 build they are Q4.12 values.
//
// Head pruning: H1 .. H6 run only for the heads head_mask keeps, in increasing order. A
// pruned head is never visited: none of its projections, scores, softmax or context is
// computed, it costs no cycles, and its 16 context channels stay zero. A block whose heads
// are all pruned goes from LN1 straight to RES1.
//
// Occupancy per image (cycles from accepting one image to accepting the next, when the next
// stage is ready): 7 + 2 x ceil(50 / LIN_UNITS) + kept_heads x (103 + ceil(50 / LIN_UNITS)).
//   16-bit build: 107 + 153 x kept_heads
//   INT8 build:    33 + 116 x kept_heads
//
// Handshake: an image is accepted when in_valid && in_ready (in_ready is high only in IDLE)
// and passed on when out_valid && out_ready. head_mask is sampled when the image is
// accepted, so changing it never affects an image that is already in the block.
//
// There is no output projection after attention: the concatenated head outputs are added
// to the residual directly.
module transformer_block_head_serial (
    input  logic clk,
    input  logic rst,

    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic [15:0]                  in_image_id,
    input  logic signed [`MAT_T_D_W-1:0] x_flat,

    // 16-bit weights and biases (biases and LayerNorm parameters are used in both builds).
    input  signed [`VEC_D_W-1:0]   n1w_flat,
    input  signed [`VEC_D_W-1:0]   n1b_flat,
    input  signed [`W_D_D_W-1:0]   wq_flat,
    input  signed [`VEC_D_W-1:0]   bq_flat,
    input  signed [`W_D_D_W-1:0]   wk_flat,
    input  signed [`VEC_D_W-1:0]   bk_flat,
    input  signed [`W_D_D_W-1:0]   wv_flat,
    input  signed [`VEC_D_W-1:0]   bv_flat,
    input  signed [`VEC_D_W-1:0]   n2w_flat,
    input  signed [`VEC_D_W-1:0]   n2b_flat,
    input  signed [`W_DFF_D_W-1:0] fc1_w_flat,
    input  signed [`VEC_DFF_W-1:0] fc1_b_flat,
    input  signed [`W_D_DFF_W-1:0] fc2_w_flat,
    input  signed [`VEC_D_W-1:0]   fc2_b_flat,

    // INT8 parameters of the linear layers (used only in the LINEAR_INT8 build).
    input  signed [`W8_D_D_W-1:0]   wq8_flat,
    input  signed [`W8_D_D_W-1:0]   wk8_flat,
    input  signed [`W8_D_D_W-1:0]   wv8_flat,
    input  signed [`W8_DFF_D_W-1:0] fc1_w8_flat,
    input  signed [`W8_D_DFF_W-1:0] fc2_w8_flat,
    input  logic [31:0]             xm_q, xm_k, xm_v, xm_fc1, xm_fc2,
    input  logic [`MULT_D_W-1:0]    ym_q_flat, ym_k_flat, ym_v_flat, ym_fc2_flat,
    input  logic [`MULT_DFF_W-1:0]  ym_fc1_flat,

    input  logic [`HEADS-1:0]      head_mask,

    output logic                         out_valid,
    input  logic                         out_ready,
    output logic [15:0]                  out_image_id,
    output logic signed [`MAT_T_D_W-1:0] y_flat,

    // High while an image is in the block.
    output logic                         busy
);

    // Bus widths for one head: Q_h and K_h at full 64-bit accumulator precision, scores
    // 32 bit, softmax intermediates 16 bit, row sums 32 bit.
    localparam int HQ_W   = `T * `HEAD_DIM * 64;
    localparam int HV_W   = `T * `HEAD_DIM * `WIDTH;
    localparam int HS_W   = `T * `T * `SCORE_WIDTH;
    localparam int HA_W   = `T * `T * `WIDTH;
    localparam int HSUM_W = `T * 32;
    // A head index, plus the value HEADS meaning "no further head".
    localparam int HIDX_W = $clog2(`HEADS + 1);
    localparam int U = `LIN_UNITS;

    `include "fixed_math.vh"
    `include "exp_lut.vh"

    typedef enum logic [3:0] {
        S_IDLE, S_LN1,
        S_H1_QKV, S_H2_SCORE, S_H3_CENTER, S_H4_EXP, S_H5_PROB, S_H6_PV,
        S_RES1, S_LN2, S_FC1, S_GELU, S_FC2, S_RES2, S_DONE
    } state_t;

    // Lowest kept head with index >= first, or HEADS when there is none.
    function automatic logic [HIDX_W-1:0] next_kept_head(input logic [`HEADS-1:0] mask,
                                                         input int first);
        next_kept_head = HIDX_W'(`HEADS);
        for (int h = `HEADS - 1; h >= first; h = h - 1)
            if (mask[h])
                next_kept_head = HIDX_W'(h);
    endfunction

    state_t state;
    logic [6:0] tok;                    // first token row of the current cycle
    logic [HIDX_W-1:0] h_cur;           // head processed by H1 .. H6
    logic [HIDX_W-1:0] first_head;      // first kept head of the image in the block
    logic [HIDX_W-1:0] next_head;       // kept head after h_cur
    logic [`HEADS-1:0] mask_r;          // head_mask of the image in the block
    logic [15:0] image_id_r;

    logic signed [`MAT_T_D_W-1:0] x_r;      // block input, also residual-1 operand
    logic signed [`MAT_T_D_W-1:0] norm1_r;
    logic signed [HQ_W-1:0]       q_h_r, k_h_r;
    logic signed [HV_W-1:0]       v_h_r;
    logic signed [HS_W-1:0]       score_h_r;
    logic signed [HA_W-1:0]       centered_h_r;
    logic signed [HA_W-1:0]       exp_h_r;
    logic [HSUM_W-1:0]            sum_exp_h_r;
    logic signed [HA_W-1:0]       prob_h_r;
    logic signed [`MAT_T_D_W-1:0] ctx_r;    // context of all heads; pruned heads stay 0
    logic signed [`MAT_T_D_W-1:0] att_out_r;
    logic signed [`MAT_T_D_W-1:0] norm2_r;
    logic signed [`MAT_T_DFF_W-1:0] fc1_r;
    logic signed [`MAT_T_DFF_W-1:0] gelu_r;
    logic signed [`MAT_T_D_W-1:0] fc2_r;
    logic signed [`MAT_T_D_W-1:0] y_r;

    logic signed [`MAT_T_D_W-1:0] norm1_comb;
    logic signed [`T*`SCORE_WIDTH-1:0] score_row_comb;      // one query row of S_h
    logic signed [HA_W-1:0]       centered_h_comb;
    logic signed [HA_W-1:0]       exp_h_comb;
    logic [HSUM_W-1:0]            sum_exp_h_comb;
    logic signed [HA_W-1:0]       prob_h_comb;
    logic signed [`HEAD_DIM*`WIDTH-1:0] ctx_row_comb;       // one row of context_h
    logic signed [`MAT_T_D_W-1:0] att_out_comb;
    logic signed [`MAT_T_D_W-1:0] norm2_comb;
    logic signed [`MAT_T_DFF_W-1:0] gelu_comb;
    logic signed [`MAT_T_D_W-1:0] y_comb;

    // First of the 64 projection / context channels that belong to head h_cur.
    int base_d;
    assign base_d = (h_cur < `HEADS) ? int'(h_cur) * `HEAD_DIM : 0;

    assign first_head = next_kept_head(mask_r, 0);
    assign next_head  = next_kept_head(mask_r, int'(h_cur) + 1);

    assign in_ready     = (state == S_IDLE);
    assign out_valid    = (state == S_DONE);
    assign out_image_id = image_id_r;
    assign y_flat       = y_r;
    assign busy         = (state != S_IDLE);

    // ------------------------------------------------------------------ linear units
    // Unit u handles token row tok + u this cycle (row 0 when that is past the last token;
    // its result is then not stored).
    logic [6:0] lin_row [U];
    logic signed [`HEAD_DIM*64-1:0]     q_row_full [U];
    logic signed [`HEAD_DIM*64-1:0]     k_row_full [U];
    logic signed [`HEAD_DIM*`WIDTH-1:0] v_row      [U];
    logic signed [`DFF*`WIDTH-1:0]      fc1_row    [U];
    logic signed [`D*`WIDTH-1:0]        fc2_row    [U];

    generate
        for (genvar u = 0; u < U; u = u + 1) begin : g_lin
            assign lin_row[u] = (int'(tok) + u < `T) ? 7'(int'(tok) + u) : 7'd0;

            // H1: the 16 channels of head h_cur of Q, K and V for one token.
            linear_row_unit #(.IN_DIM(`D), .OUT_DIM(`HEAD_DIM), .W_ROWS(`D)) u_q (
                .x_row(norm1_r[int'(lin_row[u])*`D*`WIDTH +: `D*`WIDTH]), .row_base(16'(base_d)),
                .w_flat(wq_flat), .b_flat(bq_flat), .w8_flat(wq8_flat), .x_mult(xm_q), .y_mult_flat(ym_q_flat),
                .y_full_row(q_row_full[u]), .y_row());
            linear_row_unit #(.IN_DIM(`D), .OUT_DIM(`HEAD_DIM), .W_ROWS(`D)) u_k (
                .x_row(norm1_r[int'(lin_row[u])*`D*`WIDTH +: `D*`WIDTH]), .row_base(16'(base_d)),
                .w_flat(wk_flat), .b_flat(bk_flat), .w8_flat(wk8_flat), .x_mult(xm_k), .y_mult_flat(ym_k_flat),
                .y_full_row(k_row_full[u]), .y_row());
            linear_row_unit #(.IN_DIM(`D), .OUT_DIM(`HEAD_DIM), .W_ROWS(`D)) u_v (
                .x_row(norm1_r[int'(lin_row[u])*`D*`WIDTH +: `D*`WIDTH]), .row_base(16'(base_d)),
                .w_flat(wv_flat), .b_flat(bv_flat), .w8_flat(wv8_flat), .x_mult(xm_v), .y_mult_flat(ym_v_flat),
                .y_full_row(), .y_row(v_row[u]));

            // FC1 and FC2 for one token.
            linear_row_unit #(.IN_DIM(`D), .OUT_DIM(`DFF), .W_ROWS(`DFF)) u_fc1 (
                .x_row(norm2_r[int'(lin_row[u])*`D*`WIDTH +: `D*`WIDTH]), .row_base(16'd0),
                .w_flat(fc1_w_flat), .b_flat(fc1_b_flat), .w8_flat(fc1_w8_flat), .x_mult(xm_fc1), .y_mult_flat(ym_fc1_flat),
                .y_full_row(), .y_row(fc1_row[u]));
            linear_row_unit #(.IN_DIM(`DFF), .OUT_DIM(`D), .W_ROWS(`D)) u_fc2 (
                .x_row(gelu_r[int'(lin_row[u])*`DFF*`WIDTH +: `DFF*`WIDTH]), .row_base(16'd0),
                .w_flat(fc2_w_flat), .b_flat(fc2_b_flat), .w8_flat(fc2_w8_flat), .x_mult(xm_fc2), .y_mult_flat(ym_fc2_flat),
                .y_full_row(), .y_row(fc2_row[u]));
        end
    endgenerate

    // ------------------------------------------------------------------ attention units
    // H2: query row tok of S_h: S_h[tok][j] = sum of Q_h[tok][c] * K_h[j][c] over the 16
    //   channels c. Q8.24 * Q8.24 is Q16.48: >>> 36 gives Q4.12, >>> 2 divides by
    //   sqrt(HEAD_DIM) = 4. The result is saturated to 32 bit.
    always_comb begin
        logic signed [63:0] score_acc, qv, kv;
        score_row_comb = '0;
        for (int j = 0; j < `T; j = j + 1) begin
            score_acc = 64'sd0;
            for (int c = 0; c < `HEAD_DIM; c = c + 1) begin
                qv = `ELEM_W(q_h_r, int'(tok)*`HEAD_DIM+c, 64);
                kv = `ELEM_W(k_h_r, j*`HEAD_DIM+c, 64);
                score_acc = score_acc + qv * kv;
            end
            `ELEM_W(score_row_comb, j, `SCORE_WIDTH) = sat32((score_acc >>> 36) >>> 2);
        end
    end

    // H3: subtract the row maximum, so every value is <= 0 (numerically stable softmax).
    //   Values below -8.0 (-32768) are clamped; exp(-8) is about 1 LSB.
    always_comb begin
        integer i, j;
        logic signed [63:0] maxv, sv, diff;
        centered_h_comb = '0;
        for (i = 0; i < `T; i = i + 1) begin
            maxv = $signed(`ELEM_W(score_h_r, i*`T, `SCORE_WIDTH));
            for (j = 1; j < `T; j = j + 1) begin
                sv = $signed(`ELEM_W(score_h_r, i*`T+j, `SCORE_WIDTH));
                if (sv > maxv)
                    maxv = sv;
            end
            for (j = 0; j < `T; j = j + 1) begin
                sv = $signed(`ELEM_W(score_h_r, i*`T+j, `SCORE_WIDTH));
                diff = sv - maxv;
                if (diff < -64'sd32768)
                    diff = -64'sd32768;
                `ELEM(centered_h_comb, i*`T+j) = diff[`WIDTH-1:0];
            end
        end
    end

    // H4: exp by lookup table (Q4.12, at most 1.0 = 4096) and the sum of each row.
    always_comb begin
        integer i, j;
        logic signed [`WIDTH-1:0] exp_val;
        logic [31:0] sum_val;
        exp_h_comb = '0;
        sum_exp_h_comb = '0;
        for (i = 0; i < `T; i = i + 1) begin
            sum_val = 32'd0;
            for (j = 0; j < `T; j = j + 1) begin
                exp_val = exp_lut_lookup($signed(`ELEM(centered_h_r, i*`T+j)));
                `ELEM(exp_h_comb, i*`T+j) = exp_val;
                sum_val = sum_val + {16'd0, exp_val};
            end
            `ELEM_W(sum_exp_h_comb, i, 32) = sum_val;
        end
    end

    // H5: P_h = (exp << 12) / row_sum, Q4.12 in [0, 1]. The division truncates.
    always_comb begin
        integer i, j;
        logic [31:0] num, den;
        prob_h_comb = '0;
        for (i = 0; i < `T; i = i + 1) begin
            den = `ELEM_W(sum_exp_h_r, i, 32);
            for (j = 0; j < `T; j = j + 1) begin
                num = {16'd0, `ELEM(exp_h_r, i*`T+j)} << `FRAC;
                if (den == 32'd0)
                    `ELEM(prob_h_comb, i*`T+j) = {`WIDTH{1'b0}};
                else
                    `ELEM(prob_h_comb, i*`T+j) = num / den;
            end
        end
    end

    // H6: row tok of context_h = P_h * V_h, accumulated in Q8.24 and shifted back to Q4.12.
    always_comb begin
        logic signed [63:0] acc, pv, vv;
        ctx_row_comb = '0;
        for (int c = 0; c < `HEAD_DIM; c = c + 1) begin
            acc = 64'sd0;
            for (int t = 0; t < `T; t = t + 1) begin
                pv = $signed(`ELEM(prob_h_r, int'(tok)*`T+t));
                vv = $signed(`ELEM(v_h_r, t*`HEAD_DIM+c));
                acc = acc + pv * vv;
            end
            `ELEM(ctx_row_comb, c) = sat16(acc >>> `FRAC);
        end
    end

    // ------------------------------------------------------------------ element-wise steps
    norm_stage_50x64 u_norm1 (.x_flat(x_r), .w_flat(n1w_flat), .b_flat(n1b_flat), .y_flat(norm1_comb));
    residual_add_50x64 u_residual1 (.a_flat(x_r), .b_flat(ctx_r), .y_flat(att_out_comb));
    norm_stage_50x64 u_norm2 (.x_flat(att_out_r), .w_flat(n2w_flat), .b_flat(n2b_flat), .y_flat(norm2_comb));
    gelu_stage_50x128 u_gelu (.x_flat(fc1_r), .y_flat(gelu_comb));
    residual_add_50x64 u_residual2 (.a_flat(att_out_r), .b_flat(fc2_r), .y_flat(y_comb));

    // ------------------------------------------------------------------ state machine
    // Each state stores the result of its micro-step. Token-serial states advance tok and
    // move on after the last token row.
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            tok <= '0;
            h_cur <= '0;
            mask_r <= '0;
            image_id_r <= 16'd0;
            x_r <= '0;
            norm1_r <= '0;
            q_h_r <= '0; k_h_r <= '0; v_h_r <= '0;
            score_h_r <= '0; centered_h_r <= '0;
            exp_h_r <= '0; sum_exp_h_r <= '0; prob_h_r <= '0;
            ctx_r <= '0;
            att_out_r <= '0;
            norm2_r <= '0;
            fc1_r <= '0;
            gelu_r <= '0;
            fc2_r <= '0;
            y_r <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (in_valid) begin
                        x_r <= x_flat;
                        image_id_r <= in_image_id;
                        mask_r <= head_mask;
                        ctx_r <= '0;
                        tok <= '0;
                        state <= S_LN1;
                    end
                end
                S_LN1: begin
                    norm1_r <= norm1_comb;
                    // First kept head, or straight to residual 1 if every head of this
                    // block is pruned.
                    h_cur <= first_head;
                    state <= (first_head < `HEADS) ? S_H1_QKV : S_RES1;
                end
                S_H1_QKV: begin
                    for (int u = 0; u < U; u = u + 1)
                        if (int'(tok) + u < `T) begin
                            q_h_r[(int'(tok)+u)*`HEAD_DIM*64 +: `HEAD_DIM*64] <= q_row_full[u];
                            k_h_r[(int'(tok)+u)*`HEAD_DIM*64 +: `HEAD_DIM*64] <= k_row_full[u];
                            v_h_r[(int'(tok)+u)*`HEAD_DIM*`WIDTH +: `HEAD_DIM*`WIDTH] <= v_row[u];
                        end
                    if (int'(tok) + U >= `T) begin
                        tok <= '0;
                        state <= S_H2_SCORE;
                    end else begin
                        tok <= tok + 7'(U);
                    end
                end
                S_H2_SCORE: begin
                    score_h_r[int'(tok)*`T*`SCORE_WIDTH +: `T*`SCORE_WIDTH] <= score_row_comb;
                    if (int'(tok) == `T - 1) begin
                        tok <= '0;
                        state <= S_H3_CENTER;
                    end else begin
                        tok <= tok + 7'd1;
                    end
                end
                S_H3_CENTER: begin
                    centered_h_r <= centered_h_comb;
                    state <= S_H4_EXP;
                end
                S_H4_EXP: begin
                    exp_h_r <= exp_h_comb;
                    sum_exp_h_r <= sum_exp_h_comb;
                    state <= S_H5_PROB;
                end
                S_H5_PROB: begin
                    prob_h_r <= prob_h_comb;
                    state <= S_H6_PV;
                end
                S_H6_PV: begin
                    ctx_r[(int'(tok)*`D + base_d)*`WIDTH +: `HEAD_DIM*`WIDTH] <= ctx_row_comb;
                    if (int'(tok) == `T - 1) begin
                        tok <= '0;
                        // Next kept head, or on to residual 1 after the last one.
                        h_cur <= next_head;
                        state <= (next_head < `HEADS) ? S_H1_QKV : S_RES1;
                    end else begin
                        tok <= tok + 7'd1;
                    end
                end
                S_RES1: begin
                    att_out_r <= att_out_comb;
                    state <= S_LN2;
                end
                S_LN2: begin
                    norm2_r <= norm2_comb;
                    state <= S_FC1;
                end
                S_FC1: begin
                    for (int u = 0; u < U; u = u + 1)
                        if (int'(tok) + u < `T)
                            fc1_r[(int'(tok)+u)*`DFF*`WIDTH +: `DFF*`WIDTH] <= fc1_row[u];
                    if (int'(tok) + U >= `T) begin
                        tok <= '0;
                        state <= S_GELU;
                    end else begin
                        tok <= tok + 7'(U);
                    end
                end
                S_GELU: begin
                    gelu_r <= gelu_comb;
                    state <= S_FC2;
                end
                S_FC2: begin
                    for (int u = 0; u < U; u = u + 1)
                        if (int'(tok) + u < `T)
                            fc2_r[(int'(tok)+u)*`D*`WIDTH +: `D*`WIDTH] <= fc2_row[u];
                    if (int'(tok) + U >= `T) begin
                        tok <= '0;
                        state <= S_RES2;
                    end else begin
                        tok <= tok + 7'(U);
                    end
                end
                S_RES2: begin
                    y_r <= y_comb;
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
