// ============================================================================
// quantize.sv - INT32 -> INT8 requantization (right-shift + saturate)
// ============================================================================

module quantize
    import npu_types_pkg::*;
(
    input  logic                       clk,
    input  logic                       rst_n,

    // Quantization parameter
    input  logic [4:0]                 quant_shift,

    // Stream in  (INT32)
    input  logic signed [ACC_W-1:0]    in_data,
    input  logic                       in_valid,
    output logic                       in_ready,

    // Stream out (INT8)
    output logic signed [DATA_W-1:0]   out_data,
    output logic                       out_valid,
    input  logic                       out_ready
);

    localparam logic signed [ACC_W-1:0] SAT_MAX =  32'sd127;
    localparam logic signed [ACC_W-1:0] SAT_MIN = -32'sd128;

    logic signed [ACC_W-1:0]  shifted;
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [ACC_W-1:0]  clamped;
    /* verilator lint_on UNUSEDSIGNAL */
    logic signed [DATA_W-1:0] quantized;

    assign shifted   = in_data >>> quant_shift;  // arithmetic right shift
    assign clamped   = (shifted > SAT_MAX) ? SAT_MAX :
                       (shifted < SAT_MIN) ? SAT_MIN : shifted;
    assign quantized = clamped[DATA_W-1:0];

    pipeline_reg #(.DATA_W(DATA_W)) u_reg (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_data   (quantized),
        .in_valid  (in_valid),
        .in_ready  (in_ready),
        .out_data  (out_data),
        .out_valid (out_valid),
        .out_ready (out_ready)
    );

endmodule : quantize
