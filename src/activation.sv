// ============================================================================
// activation.sv - Configurable activation on INT32
//   ACT_RELU (default): max(0, x)
//   ACT_NONE:           passthrough
//   ACT_LEAKY_RELU:     x >= 0 ? x : x >>> LEAKY_SHIFT
// ============================================================================

module activation
    import npu_types_pkg::*;
(
    input  logic                       clk,
    input  logic                       rst_n,

    // Configuration
    input  act_mode_e                  act_mode,

    // Stream in  (INT32)
    input  logic signed [ACC_W-1:0]    in_data,
    input  logic                       in_valid,
    output logic                       in_ready,

    // Stream out (INT32, activation applied)
    output logic signed [ACC_W-1:0]    out_data,
    output logic                       out_valid,
    input  logic                       out_ready
);

    logic signed [ACC_W-1:0] activated;

    always_comb begin
        case (act_mode)
            ACT_NONE:       activated = in_data;
            ACT_LEAKY_RELU: activated = (in_data < 0)
                                        ? (in_data >>> LEAKY_SHIFT)
                                        : in_data;
            default:        activated = (in_data < 0) ? '0 : in_data;
        endcase
    end

    pipeline_reg #(.DATA_W(ACC_W)) u_reg (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_data   (activated),
        .in_valid  (in_valid),
        .in_ready  (in_ready),
        .out_data  (out_data),
        .out_valid (out_valid),
        .out_ready (out_ready)
    );

endmodule : activation
