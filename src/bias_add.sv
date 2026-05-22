// ============================================================================
// bias_add.sv - Add sign-extended INT8 bias to INT32 accumulator
// ============================================================================

module bias_add
    import npu_types_pkg::*;
(
    input  logic                       clk,
    input  logic                       rst_n,

    // Bias value (INT8, sign-extended to ACC_W by this module)
    input  logic signed [DATA_W-1:0]   bias_val,

    // Stream in  (INT32 accumulator)
    input  logic signed [ACC_W-1:0]    in_data,
    input  logic                       in_valid,
    output logic                       in_ready,

    // Stream out (INT32 with bias added)
    output logic signed [ACC_W-1:0]    out_data,
    output logic                       out_valid,
    input  logic                       out_ready
);

    logic signed [ACC_W-1:0] biased;
    assign biased = in_data + ACC_W'(bias_val);

    pipeline_reg #(.DATA_W(ACC_W)) u_reg (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_data   (biased),
        .in_valid  (in_valid),
        .in_ready  (in_ready),
        .out_data  (out_data),
        .out_valid (out_valid),
        .out_ready (out_ready)
    );

endmodule : bias_add
