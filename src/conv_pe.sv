// ============================================================================
// conv_pe.sv - Single processing element: INT8xINT8 -> INT32 MAC
// ============================================================================

module conv_pe
    import npu_types_pkg::*;
(
    input  logic                      clk,
    input  logic                      rst_n,

    // Operands
    input  logic signed [DATA_W-1:0]  act_in,
    input  logic signed [DATA_W-1:0]  wt_in,
    input  logic signed [ACC_W-1:0]   acc_in,

    // Result
    output logic signed [ACC_W-1:0]   acc_out,

    // Handshake
    input  logic                      valid_in,
    output logic                      ready_out,
    output logic                      valid_out,
    input  logic                      ready_in,

    // Combinational signed-overflow flag on the internal add
    // (acc_in + sign_extend(act_in * wt_in)). Valid only on cycles where
    // the caller intends to accept this product (i.e. valid_in & ready_out).
    output logic                      acc_overflow
);

    logic signed [2*DATA_W-1:0]       mul;
    logic signed [ACC_W-1:0]          product;
    logic signed [ACC_W-1:0]          result_r;
    logic                             valid_r;
    logic                             do_accept, do_emit;

    assign mul       = act_in   * wt_in;
    assign product   = acc_in   + ACC_W'(mul);
    assign do_accept = valid_in & ready_out;
    assign do_emit   = valid_r  & ready_in;
    assign ready_out = ~valid_r | ready_in;
    assign acc_out   = result_r;
    assign valid_out = valid_r;

    // Signed addition overflows iff both addends share a sign and the sum
    // takes the opposite sign. The sign-extended product's sign is mul[15].
    assign acc_overflow = (acc_in[ACC_W-1]   == mul[2*DATA_W-1]) &&
                          (product[ACC_W-1]  != acc_in[ACC_W-1]);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_r <= '0;
            valid_r  <= 1'b0;
        end else begin
            if (do_accept) begin
                result_r <= product;
                valid_r  <= 1'b1;
            end else if (do_emit) begin
                valid_r <= 1'b0;
            end
        end
    end

endmodule : conv_pe
