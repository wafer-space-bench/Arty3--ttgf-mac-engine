// ============================================================================
// regfile.sv - 8-bit register file for the SPI-controlled MAC engine
//
// See docs/SPEC.md section 4 (register map), section 5.4 (read side-effects),
// and section 8.2 (reset state).
//
// Owns:
//   - The configuration registers (OP_A, OP_B, BIAS, QUANT_SHIFT, ACT_MODE)
//   - The output register RESULT and its valid flag
//   - The sticky overflow flag ACC_OVF_STK
//   - The 32-bit accumulator shadow register
//
// Hard reset (rst_n low) clears all storage.
// Soft reset (soft_reset, pulsed by cmd_fsm on CMD_RESET) clears RESULT,
// RESULT_VALID, ACC_OVF_STK, and the accumulator shadow but preserves
// OP_A / OP_B / BIAS / QUANT_SHIFT / ACT_MODE.
// ============================================================================

`default_nettype none

module regfile import npu_types_pkg::*; (
    input  logic                       clk,
    input  logic                       rst_n,        // hard reset
    input  logic                       soft_reset,   // 1-cycle pulse from cmd_fsm

    // SPI-slave port
    input  logic [6:0]                 rf_addr,
    input  logic [7:0]                 rf_wdata,
    input  logic                       rf_we,
    input  logic                       rf_re,
    output logic [7:0]                 rf_rdata,

    // To cmd_fsm
    output logic                       cmd_valid,    // 1-cycle pulse on CMD write
    output logic [7:0]                 cmd_code,
    input  logic [2:0]                 fsm_state,

    // Configuration outputs (consumed by cmd_fsm / postproc / conv_pe)
    output logic signed [DATA_W-1:0]   op_a,
    output logic signed [DATA_W-1:0]   op_b,
    output logic signed [DATA_W-1:0]   bias_val,
    output logic [4:0]                 quant_shift_val,
    output logic [1:0]                 act_mode_bits,

    // Live accumulator tap (for ACC_B0 read snapshot)
    input  logic signed [ACC_W-1:0]    pe_acc_out,

    // Post-processing completion (from cmd_fsm)
    input  logic                       pp_done,
    input  logic signed [DATA_W-1:0]   pp_data,

    // MAC overflow indication (from cmd_fsm)
    input  logic                       mac_overflow,

    // Status flags exported to the uo_out status mirror (sec 7.2)
    output logic                       status_idle,
    output logic                       status_busy,
    output logic                       status_result_valid,
    output logic                       status_acc_ovf_stk
);
    // Register address map (sec 4)

    localparam logic [6:0] A_STATUS      = 7'h00;
    localparam logic [6:0] A_CMD         = 7'h01;
    localparam logic [6:0] A_OP_A        = 7'h02;
    localparam logic [6:0] A_OP_B        = 7'h03;
    localparam logic [6:0] A_BIAS        = 7'h04;
    localparam logic [6:0] A_QUANT_SHIFT = 7'h05;
    localparam logic [6:0] A_ACT_MODE    = 7'h06;
    localparam logic [6:0] A_ACC_B0      = 7'h08;
    localparam logic [6:0] A_ACC_B1      = 7'h09;
    localparam logic [6:0] A_ACC_B2      = 7'h0A;
    localparam logic [6:0] A_ACC_B3      = 7'h0B;
    localparam logic [6:0] A_RESULT      = 7'h0C;
    localparam logic [6:0] A_FEATURE_ID  = 7'h10;

    localparam logic [7:0] FEATURE_ID_VAL = 8'hA1;

    // Storage

    logic signed [DATA_W-1:0] op_a_r;
    logic signed [DATA_W-1:0] op_b_r;
    logic signed [DATA_W-1:0] bias_r;
    logic [4:0]               quant_shift_r;
    logic [1:0]               act_mode_r;

    logic signed [DATA_W-1:0] result_r;
    logic                     result_valid_r;
    logic                     acc_ovf_stk_r;
    logic signed [ACC_W-1:0]  acc_shadow_r;

    // Writes, side-effects, soft reset

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            op_a_r         <= '0;
            op_b_r         <= '0;
            bias_r         <= '0;
            quant_shift_r  <= 5'd0;
            act_mode_r     <= 2'b00;
            result_r       <= '0;
            result_valid_r <= 1'b0;
            acc_ovf_stk_r  <= 1'b0;
            acc_shadow_r   <= '0;
        end else if (soft_reset) begin
            // CMD_RESET: clear accumulator-related state; preserve config
            result_r       <= '0;
            result_valid_r <= 1'b0;
            acc_ovf_stk_r  <= 1'b0;
            acc_shadow_r   <= '0;
        end else begin
            // Sticky overflow set on any MAC that overflows the INT32 acc
            if (mac_overflow) begin
                acc_ovf_stk_r <= 1'b1;
            end

            // Post-processing completion latches RESULT and sets RESULT_VALID.
            // Reading RESULT clears RESULT_VALID. If both fire on the same
            // cycle, the pp_done path wins (RESULT_VALID stays 1, with the
            // freshly-latched RESULT value).
            if (pp_done) begin
                result_r       <= pp_data;
                result_valid_r <= 1'b1;
            end else if (rf_re && (rf_addr == A_RESULT)) begin
                result_valid_r <= 1'b0;
            end

            // Accumulator shadow load on ACC_B0 read (sec 5.4).
            // Same-cycle race vs end-of-MAC writeback: the shadow samples
            // the pre-MAC value of pe_acc_out (the PE's result_r updates on
            // this same clk edge; the value we see here is the old one).
            if (rf_re && (rf_addr == A_ACC_B0)) begin
                acc_shadow_r <= pe_acc_out;
            end

            // Storage writes (CMD is dispatched separately via cmd_valid)
            if (rf_we) begin
                unique case (rf_addr)
                    A_OP_A:        op_a_r        <= rf_wdata;
                    A_OP_B:        op_b_r        <= rf_wdata;
                    A_BIAS:        bias_r        <= rf_wdata;
                    A_QUANT_SHIFT: quant_shift_r <= rf_wdata[4:0];
                    A_ACT_MODE:    act_mode_r    <= rf_wdata[1:0];
                    default: ;  // CMD, RO, and unmapped: no storage update
                endcase
            end
        end
    end

    // Combinational outputs

    // Configuration taps
    assign op_a            = op_a_r;
    assign op_b            = op_b_r;
    assign bias_val        = bias_r;
    assign quant_shift_val = quant_shift_r;
    assign act_mode_bits   = act_mode_r;

    // Status (mirrored to uo_out)
    assign status_idle         = (fsm_state == 3'b000);
    assign status_busy         = ~status_idle;
    assign status_result_valid = result_valid_r;
    assign status_acc_ovf_stk  = acc_ovf_stk_r;

    // CMD write -> 1-cycle command pulse. cmd_fsm gates on state == IDLE
    // (see sec 4.2 for the boundary rule).
    assign cmd_valid = rf_we & (rf_addr == A_CMD);
    assign cmd_code  = rf_wdata;

    // Read mux. ACC_B0 returns the live LSB of pe_acc_out (which is the same
    // value that gets latched into the shadow on this very cycle, so the
    // four-byte read is coherent provided the host reads B0 first).
    logic [7:0] status_byte;
    assign status_byte = {4'b0000,
                          status_acc_ovf_stk,
                          status_result_valid,
                          status_busy,
                          status_idle};

    always_comb begin
        unique case (rf_addr)
            A_STATUS:      rf_rdata = status_byte;
            A_OP_A:        rf_rdata = op_a_r;
            A_OP_B:        rf_rdata = op_b_r;
            A_BIAS:        rf_rdata = bias_r;
            A_QUANT_SHIFT: rf_rdata = {3'b000, quant_shift_r};
            A_ACT_MODE:    rf_rdata = {6'b000000, act_mode_r};
            A_ACC_B0:      rf_rdata = pe_acc_out[ 7: 0];
            A_ACC_B1:      rf_rdata = acc_shadow_r[15: 8];
            A_ACC_B2:      rf_rdata = acc_shadow_r[23:16];
            A_ACC_B3:      rf_rdata = acc_shadow_r[31:24];
            A_RESULT:      rf_rdata = result_r;
            A_FEATURE_ID:  rf_rdata = FEATURE_ID_VAL;
            default:       rf_rdata = 8'h00;
        endcase
    end

endmodule : regfile
