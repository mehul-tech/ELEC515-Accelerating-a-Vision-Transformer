// Saturate a 64-bit intermediate to the signed 16-bit range.
function automatic logic signed [`WIDTH-1:0] sat16(input logic signed [63:0] x);
    if (x > 64'sd32767)
        sat16 = 16'sd32767;
    else if (x < -64'sd32768)
        sat16 = -16'sd32768;
    else
        sat16 = x[`WIDTH-1:0];
endfunction

// Saturate a 64-bit intermediate to the signed 32-bit range.
function automatic logic signed [31:0] sat32(input logic signed [63:0] x);
    if (x > 64'sd2147483647)
        sat32 = 32'sh7fffffff;
    else if (x < -64'sd2147483648)
        sat32 = 32'sh80000000;
    else
        sat32 = x[31:0];
endfunction

// Integer square root floor(sqrt(x_in)), computed digit by digit (two bits per iteration).
function automatic logic [31:0] isqrt(input logic [63:0] x_in);
    logic [63:0] op;
    logic [63:0] res;
    logic [63:0] one;
    op  = x_in;
    res = 64'd0;
    one = 64'h4000000000000000;
    while (one > op)
        one = one >> 2;
    while (one != 0) begin
        if (op >= res + one) begin
            op  = op - (res + one);
            res = res + (one << 1);
        end
        res = res >> 1;
        one = one >> 2;
    end
    isqrt = res[31:0];
endfunction
