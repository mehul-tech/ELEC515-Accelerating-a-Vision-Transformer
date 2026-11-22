`timescale 1ns/1ps
`include "fixed_params.vh"

// Testbench of the ViT accelerator.
//
// It drives the head mask of the selected case into the accelerator, sends the exported test
// images through the valid/ready handshake, and checks every result against the golden
// outputs of the bit-exact Python model (python/vit_fixed.py): the predicted class and all
// ten logits must be identical. It also measures the speed in clock cycles.
//
// Run-time option (plusarg):
//   +CASE=<name>   use head mask head_masks/<name>.txt and the golden outputs data/<name>/
//                  (data/int8/<name>/ in the LINEAR_INT8 build). Default: keep_all.
//
// Files read (all written by python/export_rtl_data.py):
//   data/sample_count.txt       number of images
//   data/embedding_input.txt    one image per line: 50 x 64 Q4.12 values, row-major
//   data/label.txt              MNIST label of each image
//   <golden>/golden_logits.txt  10 logits per image
//   <golden>/golden_pred.txt    predicted class per image
//   head_masks/<name>.txt       6 rows (blocks) x 4 columns (heads), 1 = keep
module tb_vit;

    localparam integer MAX_SAMPLES = 1000;
    // The simulation stops with TIMEOUT when no result arrives for this many cycles (the
    // provided design delivers its first result after 4311 cycles, the next ones every 719).
    localparam integer TIMEOUT_CYCLES = 100000;

`ifdef LINEAR_INT8
    localparam string BUILD_NAME = "LINEAR_INT8";
    localparam string GOLDEN_ROOT = "./data/int8";
`else
    localparam string BUILD_NAME = "16-bit";
    localparam string GOLDEN_ROOT = "./data";
`endif

    logic clk;
    logic rst;
    logic [`HEADS-1:0] head_mask [0:`BLOCKS-1];
    logic in_valid;
    logic in_ready;
    logic [15:0] in_image_id;
    logic signed [`MAT_T_D_W-1:0] x_flat;
    logic out_valid;
    logic out_ready;
    logic [15:0] out_image_id;
    logic signed [`VEC_LOGIT_W-1:0] logits_flat;
    logic [3:0] pred_class;
    logic [6:0] stage_busy;

    string case_name, mask_file, golden_dir;

    logic signed [`MAT_T_D_W-1:0]   sample_x [0:MAX_SAMPLES-1];
    logic signed [`VEC_LOGIT_W-1:0] golden_logits [0:MAX_SAMPLES-1];
    integer golden_pred [0:MAX_SAMPLES-1];
    integer labels [0:MAX_SAMPLES-1];

    integer sample_count;
    integer send_idx, recv_count;
    integer golden_mismatch, duplicate_count, label_correct;
    longint max_logit_diff, diff;
    integer cycles, last_progress;
    bit     timed_out;
    logic   in_fire, out_fire;

    // Clock cycle (number of rising edges so far) at which each image was accepted and at
    // which its result became valid.
    integer send_cycle [0:MAX_SAMPLES-1];
    integer recv_cycle [0:MAX_SAMPLES-1];
    bit     received   [0:MAX_SAMPLES-1];

    vit_top dut (
        .clk(clk),
        .rst(rst),
        .head_mask(head_mask),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_image_id(in_image_id),
        .x_flat(x_flat),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_image_id(out_image_id),
        .logits_flat(logits_flat),
        .pred_class(pred_class),
        .stage_busy(stage_busy)
    );

    // 10 ns clock period.
    always #5 clk = ~clk;

    // A missing or short input file stops the simulation at once ($fatal, exit status 1).
    task automatic open_file(input string path, output integer fd);
        fd = $fopen(path, "r");
        if (fd == 0)
            $fatal(1, "cannot open %s (did you run python/export_rtl_data.py?)", path);
    endtask

    task automatic read_int(input integer fd, input string what, output integer val);
        integer r;
        r = $fscanf(fd, "%d", val);
        if (r != 1)
            $fatal(1, "cannot read %s (the file is shorter than expected; export again)", what);
    endtask

    task automatic load_head_mask;
        integer fd, val;
        open_file(mask_file, fd);
        for (int b = 0; b < `BLOCKS; b = b + 1)
            for (int h = 0; h < `HEADS; h = h + 1) begin
                read_int(fd, mask_file, val);
                head_mask[b][h] = (val != 0);
            end
        $fclose(fd);
    endtask

    task automatic load_samples;
        integer fd_count, fd_x, fd_label, fd_logits, fd_pred, val;
        open_file("./data/sample_count.txt", fd_count);
        read_int(fd_count, "sample_count", sample_count);
        $fclose(fd_count);
        if (sample_count > MAX_SAMPLES) begin
            $display("NOTE: using the first %0d of %0d images", MAX_SAMPLES, sample_count);
            sample_count = MAX_SAMPLES;
        end

        open_file("./data/embedding_input.txt", fd_x);
        open_file("./data/label.txt", fd_label);
        open_file({golden_dir, "/golden_logits.txt"}, fd_logits);
        open_file({golden_dir, "/golden_pred.txt"}, fd_pred);
        for (int s = 0; s < sample_count; s = s + 1) begin
            for (int idx = 0; idx < `T * `D; idx = idx + 1) begin
                read_int(fd_x, "embedding_input", val);
                `ELEM(sample_x[s], idx) = val;
            end
            for (int c = 0; c < `N_CLASSES; c = c + 1) begin
                read_int(fd_logits, "golden_logits", val);
                `ELEM_W(golden_logits[s], c, `LOGIT_WIDTH) = val;
            end
            read_int(fd_pred, "golden_pred", golden_pred[s]);
            read_int(fd_label, "label", labels[s]);
        end
        $fclose(fd_x);
        $fclose(fd_label);
        $fclose(fd_logits);
        $fclose(fd_pred);
    endtask

    // Largest absolute difference (in LSB) between the RTL logits and the golden logits,
    // computed in 64 bits so that no difference of two 32-bit logits can wrap to zero.
    task automatic logits_max_diff(input integer id, output longint max_diff);
        longint d;
        max_diff = 0;
        for (int c = 0; c < `N_CLASSES; c = c + 1) begin
            d = longint'($signed(`ELEM_W(logits_flat, c, `LOGIT_WIDTH)))
                - longint'($signed(`ELEM_W(golden_logits[id], c, `LOGIT_WIDTH)));
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
        end
    endtask

    initial begin
        if (!$value$plusargs("CASE=%s", case_name))
            case_name = "keep_all";
        mask_file = {"./head_masks/", case_name, ".txt"};
        golden_dir = {GOLDEN_ROOT, "/", case_name};

        clk = 1'b0;
        rst = 1'b1;
        in_valid = 1'b0;
        in_image_id = 16'd0;
        x_flat = '0;
        out_ready = 1'b1;
        load_head_mask();
        load_samples();

        send_idx = 0;
        recv_count = 0;
        golden_mismatch = 0;
        duplicate_count = 0;
        label_correct = 0;
        max_logit_diff = 0;
        cycles = 0;
        last_progress = 0;
        timed_out = 1'b0;
        for (int s = 0; s < MAX_SAMPLES; s = s + 1) received[s] = 1'b0;

        repeat (5) @(posedge clk);
        rst = 1'b0;

        $display("");
        $display("======== ViT ACCELERATOR CHECK ========");
        $display("build      = %s (%0d linear unit(s) per linear layer)", BUILD_NAME, `LIN_UNITS);
        $display("case       = %s   images = %0d", case_name, sample_count);
        $write("head mask  =");
        for (int b = 0; b < `BLOCKS; b = b + 1) begin
            $write(" ");
            for (int h = 0; h < `HEADS; h = h + 1) $write("%0d", head_mask[b][h]);
        end
        $display("   (blocks 1 .. 6, heads 0 .. 3 as in the mask file)");
        $fflush;

        // One clock per iteration. At the falling edge the handshake signals are stable until
        // the next rising edge, so the transfers that edge will make (in_fire, out_fire) are
        // decided here. Results are matched to the golden data by image_id. Every line is
        // flushed at once, so the output appears while the simulation runs.
        while (recv_count < sample_count && !timed_out) begin
            @(negedge clk);

            out_fire = out_valid && out_ready;
            if (out_fire) begin
                if (received[out_image_id]) begin
                    duplicate_count = duplicate_count + 1;
                    $display("DUPLICATE image=%0d", out_image_id);
                end
                received[out_image_id] = 1'b1;
                recv_cycle[out_image_id] = cycles;
                logits_max_diff(out_image_id, diff);
                if (diff > max_logit_diff)
                    max_logit_diff = diff;
                if (pred_class == labels[out_image_id][3:0])
                    label_correct = label_correct + 1;
                if (pred_class != golden_pred[out_image_id][3:0] || diff != 0) begin
                    golden_mismatch = golden_mismatch + 1;
                    $display("MISMATCH image=%0d label=%0d golden_pred=%0d rtl_pred=%0d logits_max_diff=%0d",
                             out_image_id, labels[out_image_id], golden_pred[out_image_id], pred_class, diff);
                end else begin
                    $display("MATCH    image=%0d label=%0d pred=%0d latency=%0d",
                             out_image_id, labels[out_image_id], pred_class,
                             recv_cycle[out_image_id] - send_cycle[out_image_id]);
                end
                $fflush;
                recv_count = recv_count + 1;
                last_progress = cycles;
            end
            if (cycles - last_progress > TIMEOUT_CYCLES) begin
                timed_out = 1'b1;
                $display("TIMEOUT: no result for %0d cycles; the design seems to be stuck", TIMEOUT_CYCLES);
            end

            if (send_idx < sample_count) begin
                in_valid = 1'b1;
                in_image_id = send_idx[15:0];
                x_flat = sample_x[send_idx];
            end else begin
                in_valid = 1'b0;
            end
            in_fire = in_valid && in_ready;

            @(posedge clk);
            cycles = cycles + 1;
            if (in_fire) begin
                send_cycle[send_idx] = cycles;
                send_idx = send_idx + 1;
            end
        end

        report();
        $finish;
    end

    // Summary: correctness against the golden model, then speed.
    task automatic report;
        integer lat, lat_min, lat_max, lat_sum, first_recv, last_recv;
        lat_min = 32'h7fffffff;
        lat_max = 0;
        lat_sum = 0;
        first_recv = 32'h7fffffff;
        last_recv = 0;
        for (int s = 0; s < sample_count; s = s + 1) begin
            if (received[s]) begin
                lat = recv_cycle[s] - send_cycle[s];
                lat_sum = lat_sum + lat;
                if (lat < lat_min) lat_min = lat;
                if (lat > lat_max) lat_max = lat;
                if (recv_cycle[s] < first_recv) first_recv = recv_cycle[s];
                if (recv_cycle[s] > last_recv) last_recv = recv_cycle[s];
            end
        end

        $display("");
        $display("======== SUMMARY ========");
        $display("received_images             = %0d / %0d", recv_count, sample_count);
        $display("golden_mismatch_count       = %0d", golden_mismatch);
        $display("golden_logits_max_abs_diff  = %0d", max_logit_diff);
        $display("duplicate_results           = %0d", duplicate_count);
        $display("label_accuracy              = %0d / %0d", label_correct, recv_count);
        if (received[0])
            $display("latency_single_image        = %0d cycles", recv_cycle[0] - send_cycle[0]);
        if (recv_count > 0)
            $display("latency_all_images          = min %0d  avg %0.1f  max %0d cycles",
                     lat_min, real'(lat_sum) / recv_count, lat_max);
        if (recv_count > 1)
            $display("cycles_between_results      = %0.1f", real'(last_recv - first_recv) / (recv_count - 1));
        if (recv_count == sample_count && golden_mismatch == 0 && duplicate_count == 0)
            $display("RESULT: PASS (RTL matches the Python fixed-point model bit-exactly)");
        else
            $display("RESULT: FAIL");
        $fflush;
    endtask

endmodule
