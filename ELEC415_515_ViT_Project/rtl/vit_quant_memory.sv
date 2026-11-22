`include "fixed_params.vh"

// INT8 parameters of the linear layers, read from QUANT_ROOT at simulation start in the
// LINEAR_INT8 build (all zero, and unused, in the 16-bit build). They are written by
// python/export_rtl_data.py --int8. For every block N and layer L in wq, wk, wv, fc1, fc2:
//   QUANT_ROOT/blockN/L_w8.txt      int8 weights, one per line, row-major [out][in]
//   QUANT_ROOT/blockN/L_x_mult.txt  activation multiplier x_mult (one integer)
//   QUANT_ROOT/blockN/L_y_mult.txt  output multipliers y_mult, one per output channel
// See linear_row_unit.sv for how they are used. Biases stay in the 16-bit weight files.
// File reading ($fopen / $fscanf) is supported in simulation only.
module vit_quant_memory #(
    parameter string QUANT_ROOT = "./quant"
) (
    output logic signed [`W8_D_D_W-1:0]   wq8_flat   [0:`BLOCKS-1],
    output logic signed [`W8_D_D_W-1:0]   wk8_flat   [0:`BLOCKS-1],
    output logic signed [`W8_D_D_W-1:0]   wv8_flat   [0:`BLOCKS-1],
    output logic signed [`W8_DFF_D_W-1:0] fc1_w8_flat [0:`BLOCKS-1],
    output logic signed [`W8_D_DFF_W-1:0] fc2_w8_flat [0:`BLOCKS-1],
    output logic [31:0]                   xm_q   [0:`BLOCKS-1],
    output logic [31:0]                   xm_k   [0:`BLOCKS-1],
    output logic [31:0]                   xm_v   [0:`BLOCKS-1],
    output logic [31:0]                   xm_fc1 [0:`BLOCKS-1],
    output logic [31:0]                   xm_fc2 [0:`BLOCKS-1],
    output logic [`MULT_D_W-1:0]          ym_q_flat   [0:`BLOCKS-1],
    output logic [`MULT_D_W-1:0]          ym_k_flat   [0:`BLOCKS-1],
    output logic [`MULT_D_W-1:0]          ym_v_flat   [0:`BLOCKS-1],
    output logic [`MULT_DFF_W-1:0]        ym_fc1_flat [0:`BLOCKS-1],
    output logic [`MULT_D_W-1:0]          ym_fc2_flat [0:`BLOCKS-1]
);

    // Values of the file being read; large enough for the biggest tensor (DFF x D).
    integer vals [0:`DFF*`D-1];

    task automatic read_ints(input string path, input integer n);
        integer fd, r, idx;
        fd = $fopen(path, "r");
        if (fd == 0) $fatal(1, "cannot open %s (did you run python/export_rtl_data.py?)", path);
        for (idx = 0; idx < n; idx = idx + 1) begin
            r = $fscanf(fd, "%d", vals[idx]);
            if (r != 1) $fatal(1, "cannot read %s at index %0d (export again)", path, idx);
        end
        $fclose(fd);
    endtask

    initial begin
        for (int blk = 0; blk < `BLOCKS; blk = blk + 1) begin
            wq8_flat[blk] = '0; wk8_flat[blk] = '0; wv8_flat[blk] = '0;
            fc1_w8_flat[blk] = '0; fc2_w8_flat[blk] = '0;
            xm_q[blk] = '0; xm_k[blk] = '0; xm_v[blk] = '0; xm_fc1[blk] = '0; xm_fc2[blk] = '0;
            ym_q_flat[blk] = '0; ym_k_flat[blk] = '0; ym_v_flat[blk] = '0;
            ym_fc1_flat[blk] = '0; ym_fc2_flat[blk] = '0;
        end
`ifdef LINEAR_INT8
        for (int blk = 0; blk < `BLOCKS; blk = blk + 1) begin
            string base;
            base = $sformatf("%s/block%0d", QUANT_ROOT, blk + 1);

            read_ints({base, "/wq_w8.txt"}, `D * `D);
            for (int i = 0; i < `D * `D; i = i + 1) wq8_flat[blk][i*8 +: 8] = vals[i][7:0];
            read_ints({base, "/wk_w8.txt"}, `D * `D);
            for (int i = 0; i < `D * `D; i = i + 1) wk8_flat[blk][i*8 +: 8] = vals[i][7:0];
            read_ints({base, "/wv_w8.txt"}, `D * `D);
            for (int i = 0; i < `D * `D; i = i + 1) wv8_flat[blk][i*8 +: 8] = vals[i][7:0];
            read_ints({base, "/fc1_w8.txt"}, `DFF * `D);
            for (int i = 0; i < `DFF * `D; i = i + 1) fc1_w8_flat[blk][i*8 +: 8] = vals[i][7:0];
            read_ints({base, "/fc2_w8.txt"}, `D * `DFF);
            for (int i = 0; i < `D * `DFF; i = i + 1) fc2_w8_flat[blk][i*8 +: 8] = vals[i][7:0];

            read_ints({base, "/wq_x_mult.txt"}, 1);  xm_q[blk] = vals[0];
            read_ints({base, "/wk_x_mult.txt"}, 1);  xm_k[blk] = vals[0];
            read_ints({base, "/wv_x_mult.txt"}, 1);  xm_v[blk] = vals[0];
            read_ints({base, "/fc1_x_mult.txt"}, 1); xm_fc1[blk] = vals[0];
            read_ints({base, "/fc2_x_mult.txt"}, 1); xm_fc2[blk] = vals[0];

            read_ints({base, "/wq_y_mult.txt"}, `D);
            for (int i = 0; i < `D; i = i + 1) ym_q_flat[blk][i*32 +: 32] = vals[i];
            read_ints({base, "/wk_y_mult.txt"}, `D);
            for (int i = 0; i < `D; i = i + 1) ym_k_flat[blk][i*32 +: 32] = vals[i];
            read_ints({base, "/wv_y_mult.txt"}, `D);
            for (int i = 0; i < `D; i = i + 1) ym_v_flat[blk][i*32 +: 32] = vals[i];
            read_ints({base, "/fc1_y_mult.txt"}, `DFF);
            for (int i = 0; i < `DFF; i = i + 1) ym_fc1_flat[blk][i*32 +: 32] = vals[i];
            read_ints({base, "/fc2_y_mult.txt"}, `D);
            for (int i = 0; i < `D; i = i + 1) ym_fc2_flat[blk][i*32 +: 32] = vals[i];
        end
`endif
    end

endmodule
