// ============================================================================
// npu_types_pkg.sv - Common types used across the NPU
// ============================================================================

package npu_types_pkg;

    // Data widths
    localparam int DATA_W   = 8;   // INT8 operand width
    localparam int ACC_W    = 32;  // INT32 accumulator width
    /* verilator lint_off UNUSEDPARAM */
    localparam int BIAS_W   = 8;   // INT8 bias (sign-extended to ACC_W)
    /* verilator lint_on UNUSEDPARAM */

    // Address widths
    localparam int ADDR_W      = 16;  // Local SRAM address space (64 KB)
    localparam int MMIO_ADDR_W = 20;  // MMIO address width (1 MB)
    localparam int MMIO_DATA_W = 32;  // MMIO data bus width
    localparam int EXT_ADDR_W  = 32;  // External memory address width

    // Dimension type (spatial H, W, C, K, R, S, stride, pad)
    localparam int DIM_W = 16;
    typedef logic [DIM_W-1:0] dim_t;

    // Activation-function selector
    typedef enum logic [1:0]
	{
        ACT_NONE       = 2'b00,
        ACT_RELU       = 2'b01,
        ACT_LEAKY_RELU = 2'b10
    }   act_mode_e;

    // Leaky ReLU negative-slope shift (alpha = 1/2^LEAKY_SHIFT ~ 0.125)
    localparam int LEAKY_SHIFT = 3;

    // Data-type enum (extensible for future types)
    typedef enum logic [2:0]
	{
        DTYPE_INT8 = 3'b000
    }   dtype_e;

endpackage : npu_types_pkg
