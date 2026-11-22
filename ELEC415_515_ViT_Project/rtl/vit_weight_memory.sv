`include "fixed_params.vh"

// Reads all Q4.12 weights from WEIGHT_ROOT at simulation start and drives them as constant
// buses. Each file holds one integer per line; matrices are row-major [out][in], as written
// by python/export_rtl_data.py.
// File reading ($fopen / $fscanf) is supported in simulation only; a synthesizable design
// keeps the weights in ROM or block RAM instead.
module vit_weight_memory #(
    parameter string WEIGHT_ROOT = "./weights"
) (
    output logic signed [`VEC_D_W-1:0]   n1w_flat     [0:`BLOCKS-1],
    output logic signed [`VEC_D_W-1:0]   n1b_flat     [0:`BLOCKS-1],
    output logic signed [`W_D_D_W-1:0]   wq_flat      [0:`BLOCKS-1],
    output logic signed [`VEC_D_W-1:0]   bq_flat      [0:`BLOCKS-1],
    output logic signed [`W_D_D_W-1:0]   wk_flat      [0:`BLOCKS-1],
    output logic signed [`VEC_D_W-1:0]   bk_flat      [0:`BLOCKS-1],
    output logic signed [`W_D_D_W-1:0]   wv_flat      [0:`BLOCKS-1],
    output logic signed [`VEC_D_W-1:0]   bv_flat      [0:`BLOCKS-1],

    output logic signed [`VEC_D_W-1:0]   n2w_flat     [0:`BLOCKS-1],
    output logic signed [`VEC_D_W-1:0]   n2b_flat     [0:`BLOCKS-1],
    output logic signed [`W_DFF_D_W-1:0] fc1_w_flat   [0:`BLOCKS-1],
    output logic signed [`VEC_DFF_W-1:0] fc1_b_flat   [0:`BLOCKS-1],
    output logic signed [`W_D_DFF_W-1:0] fc2_w_flat   [0:`BLOCKS-1],
    output logic signed [`VEC_D_W-1:0]   fc2_b_flat   [0:`BLOCKS-1],

    output logic signed [`VEC_D_W-1:0]   encoder_norm_w_flat,
    output logic signed [`VEC_D_W-1:0]   encoder_norm_b_flat,
    output logic signed [`VEC_D_W-1:0]   model_norm_w_flat,
    output logic signed [`VEC_D_W-1:0]   model_norm_b_flat,

    output logic signed [`W_D_D_W-1:0]   classifier_fc1_w_flat,
    output logic signed [`VEC_D_W-1:0]   classifier_fc1_b_flat,
    output logic signed [`W_C_D_W-1:0]   classifier_fc2_w_flat,
    output logic signed [`VEC_C_W-1:0]   classifier_fc2_b_flat
);

    // Each load_* task reads one tensor of a fixed size into a flat bus.
    task automatic load_vec_d(input string path, output logic signed [`VEC_D_W-1:0] flat);
        integer fd, r, val, idx;
        begin
            flat = '0;
            fd = $fopen(path, "r");
            if (fd == 0) $fatal(1, "cannot open %s (did you run python/export_rtl_data.py?)", path);
            for (idx = 0; idx < `D; idx = idx + 1) begin
                r = $fscanf(fd, "%d", val);
                if (r != 1) $fatal(1, "cannot read %s at index %0d (export again)", path, idx);
                `ELEM(flat, idx) = val;
            end
            $fclose(fd);
        end
    endtask

    task automatic load_vec_dff(input string path, output logic signed [`VEC_DFF_W-1:0] flat);
        integer fd, r, val, idx;
        begin
            flat = '0;
            fd = $fopen(path, "r");
            if (fd == 0) $fatal(1, "cannot open %s (did you run python/export_rtl_data.py?)", path);
            for (idx = 0; idx < `DFF; idx = idx + 1) begin
                r = $fscanf(fd, "%d", val);
                if (r != 1) $fatal(1, "cannot read %s at index %0d (export again)", path, idx);
                `ELEM(flat, idx) = val;
            end
            $fclose(fd);
        end
    endtask

    task automatic load_vec_c(input string path, output logic signed [`VEC_C_W-1:0] flat);
        integer fd, r, val, idx;
        begin
            flat = '0;
            fd = $fopen(path, "r");
            if (fd == 0) $fatal(1, "cannot open %s (did you run python/export_rtl_data.py?)", path);
            for (idx = 0; idx < `N_CLASSES; idx = idx + 1) begin
                r = $fscanf(fd, "%d", val);
                if (r != 1) $fatal(1, "cannot read %s at index %0d (export again)", path, idx);
                `ELEM(flat, idx) = val;
            end
            $fclose(fd);
        end
    endtask

    task automatic load_w_d_d(input string path, output logic signed [`W_D_D_W-1:0] flat);
        integer fd, r, val, idx;
        begin
            flat = '0;
            fd = $fopen(path, "r");
            if (fd == 0) $fatal(1, "cannot open %s (did you run python/export_rtl_data.py?)", path);
            for (idx = 0; idx < `D * `D; idx = idx + 1) begin
                r = $fscanf(fd, "%d", val);
                if (r != 1) $fatal(1, "cannot read %s at index %0d (export again)", path, idx);
                `ELEM(flat, idx) = val;
            end
            $fclose(fd);
        end
    endtask

    task automatic load_w_dff_d(input string path, output logic signed [`W_DFF_D_W-1:0] flat);
        integer fd, r, val, idx;
        begin
            flat = '0;
            fd = $fopen(path, "r");
            if (fd == 0) $fatal(1, "cannot open %s (did you run python/export_rtl_data.py?)", path);
            for (idx = 0; idx < `DFF * `D; idx = idx + 1) begin
                r = $fscanf(fd, "%d", val);
                if (r != 1) $fatal(1, "cannot read %s at index %0d (export again)", path, idx);
                `ELEM(flat, idx) = val;
            end
            $fclose(fd);
        end
    endtask

    task automatic load_w_d_dff(input string path, output logic signed [`W_D_DFF_W-1:0] flat);
        integer fd, r, val, idx;
        begin
            flat = '0;
            fd = $fopen(path, "r");
            if (fd == 0) $fatal(1, "cannot open %s (did you run python/export_rtl_data.py?)", path);
            for (idx = 0; idx < `D * `DFF; idx = idx + 1) begin
                r = $fscanf(fd, "%d", val);
                if (r != 1) $fatal(1, "cannot read %s at index %0d (export again)", path, idx);
                `ELEM(flat, idx) = val;
            end
            $fclose(fd);
        end
    endtask

    task automatic load_w_c_d(input string path, output logic signed [`W_C_D_W-1:0] flat);
        integer fd, r, val, idx;
        begin
            flat = '0;
            fd = $fopen(path, "r");
            if (fd == 0) $fatal(1, "cannot open %s (did you run python/export_rtl_data.py?)", path);
            for (idx = 0; idx < `N_CLASSES * `D; idx = idx + 1) begin
                r = $fscanf(fd, "%d", val);
                if (r != 1) $fatal(1, "cannot read %s at index %0d (export again)", path, idx);
                `ELEM(flat, idx) = val;
            end
            $fclose(fd);
        end
    endtask

    // Folder layout: block1 .. block6, encoder_norm, model_norm, classifier.
    task automatic load_all_weights;
        string base;
        begin
            for (int blk = 0; blk < `BLOCKS; blk = blk + 1) begin
                base = $sformatf("%s/block%0d", WEIGHT_ROOT, blk + 1);
                load_vec_d({base, "/norm1_weight.txt"}, n1w_flat[blk]);
                load_vec_d({base, "/norm1_bias.txt"}, n1b_flat[blk]);
                load_w_d_d({base, "/wq_weight.txt"}, wq_flat[blk]);
                load_vec_d({base, "/wq_bias.txt"}, bq_flat[blk]);
                load_w_d_d({base, "/wk_weight.txt"}, wk_flat[blk]);
                load_vec_d({base, "/wk_bias.txt"}, bk_flat[blk]);
                load_w_d_d({base, "/wv_weight.txt"}, wv_flat[blk]);
                load_vec_d({base, "/wv_bias.txt"}, bv_flat[blk]);
                load_vec_d({base, "/norm2_weight.txt"}, n2w_flat[blk]);
                load_vec_d({base, "/norm2_bias.txt"}, n2b_flat[blk]);
                load_w_dff_d({base, "/fc1_weight.txt"}, fc1_w_flat[blk]);
                load_vec_dff({base, "/fc1_bias.txt"}, fc1_b_flat[blk]);
                load_w_d_dff({base, "/fc2_weight.txt"}, fc2_w_flat[blk]);
                load_vec_d({base, "/fc2_bias.txt"}, fc2_b_flat[blk]);
            end

            load_vec_d({WEIGHT_ROOT, "/encoder_norm/norm_weight.txt"}, encoder_norm_w_flat);
            load_vec_d({WEIGHT_ROOT, "/encoder_norm/norm_bias.txt"}, encoder_norm_b_flat);
            load_vec_d({WEIGHT_ROOT, "/model_norm/norm_weight.txt"}, model_norm_w_flat);
            load_vec_d({WEIGHT_ROOT, "/model_norm/norm_bias.txt"}, model_norm_b_flat);
            load_w_d_d({WEIGHT_ROOT, "/classifier/fc1_weight.txt"}, classifier_fc1_w_flat);
            load_vec_d({WEIGHT_ROOT, "/classifier/fc1_bias.txt"}, classifier_fc1_b_flat);
            load_w_c_d({WEIGHT_ROOT, "/classifier/fc2_weight.txt"}, classifier_fc2_w_flat);
            load_vec_c({WEIGHT_ROOT, "/classifier/fc2_bias.txt"}, classifier_fc2_b_flat);
        end
    endtask

    initial begin
        load_all_weights();
    end

endmodule
