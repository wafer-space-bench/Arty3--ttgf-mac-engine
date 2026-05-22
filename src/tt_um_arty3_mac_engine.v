/*
 * tt_um_arty3_mac_engine.v
 *
 * Top-level wrapper for Tiny Tapeout. Standalone INT8 MAC engine with an
 * on-chip post-processing pipeline (bias -> activation -> requantize),
 * controlled over a 4-wire SPI slave on uio[3:0]. See docs/SPEC.md for the
 * full specification.
 *
 * Pin map (sec 7):
 *   ui_in[7:0]  : reserved (lint-tied off)
 *   uo_out[0]   : STATUS.IDLE
 *   uo_out[1]   : STATUS.BUSY
 *   uo_out[2]   : STATUS.RESULT_VALID
 *   uo_out[3]   : STATUS.ACC_OVF_STK
 *   uo_out[6:4] : cmd_fsm state[2:0]
 *   uo_out[7]   : heartbeat (clk / 2^20)
 *   uio[0]      : SPI_CS_N   (in)
 *   uio[1]      : SPI_SCLK   (in)
 *   uio[2]      : SPI_MOSI   (in)
 *   uio[3]      : SPI_MISO   (out, actively driven, 0 when CS_N high)
 *   uio[7:4]    : reserved   (in)
 *
 * uio_oe = 8'b0000_1000 (only MISO drives).
 */

`default_nettype none

module tt_um_arty3_mac_engine (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the project is selected
    input  wire       clk,      // system clock
    input  wire       rst_n     // active-low asynchronous reset
);

    // ------------------------------------------------------------------------
    // Reset synchroniser (sec 8.1): 2-flop async-assert / sync-deassert.
    // ------------------------------------------------------------------------

    reg [1:0] rst_sync_ff;
    wire      rst_n_sync;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_sync_ff <= 2'b00;
        else        rst_sync_ff <= {rst_sync_ff[0], 1'b1};
    end
    assign rst_n_sync = rst_sync_ff[1];

    // SPI <-> regfile <-> cmd_fsm interconnect

    wire [6:0] rf_addr;
    wire [7:0] rf_wdata;
    wire [7:0] rf_rdata;
    wire       rf_we;
    wire       rf_re;

    wire       cmd_valid;
    wire [7:0] cmd_code;
    wire [2:0] fsm_state;

    wire signed [7:0]  op_a;
    wire signed [7:0]  op_b;
    wire signed [7:0]  bias_val;
    wire        [4:0]  quant_shift_val;
    wire        [1:0]  act_mode_bits;

    wire        mac_overflow;
    wire        pp_done;
    wire signed [7:0] pp_data;
    wire        soft_reset;

    wire        status_idle;
    wire        status_busy;
    wire        status_result_valid;
    wire        status_acc_ovf_stk;

    // conv_pe wires
    wire signed [7:0]  pe_act_in;
    wire signed [7:0]  pe_wt_in;
    wire signed [31:0] pe_acc_in;
    wire signed [31:0] pe_acc_out;
    wire        pe_valid_in;
    wire        pe_ready_in;
    wire        pe_ready_out;
    wire        pe_valid_out;
    wire        pe_acc_overflow;

    // postproc wires
    wire signed [31:0] pp_in_data;
    wire        pp_in_valid;
    wire        pp_in_ready;
    wire signed [7:0]  pp_out_data;
    wire        pp_out_valid;
    wire        pp_out_ready;

    // Core reset: rst_n_sync gated by soft_reset. Drives conv_pe and postproc
    // only, so the SPI slave / regfile / cmd_fsm keep running across a soft
    // reset and the CMD_RESET write transaction completes cleanly.
    wire core_rst_n = rst_n_sync & ~soft_reset;

    // SPI slave (sec 3, 9.1)

    wire spi_miso;

    spi_slave u_spi (
        .clk          (clk),
        .rst_n        (rst_n_sync),
        .spi_cs_n_pad (uio_in[0]),
        .spi_sclk_pad (uio_in[1]),
        .spi_mosi_pad (uio_in[2]),
        .spi_miso     (spi_miso),
        .rf_addr      (rf_addr),
        .rf_wdata     (rf_wdata),
        .rf_we        (rf_we),
        .rf_re        (rf_re),
        .rf_rdata     (rf_rdata)
    );

    // Register file (sec 4)

    regfile u_rf (
        .clk                 (clk),
        .rst_n               (rst_n_sync),
        .soft_reset          (soft_reset),
        .rf_addr             (rf_addr),
        .rf_wdata            (rf_wdata),
        .rf_we               (rf_we),
        .rf_re               (rf_re),
        .rf_rdata            (rf_rdata),
        .cmd_valid           (cmd_valid),
        .cmd_code            (cmd_code),
        .fsm_state           (fsm_state),
        .op_a                (op_a),
        .op_b                (op_b),
        .bias_val            (bias_val),
        .quant_shift_val     (quant_shift_val),
        .act_mode_bits       (act_mode_bits),
        .pe_acc_out          (pe_acc_out),
        .pp_done             (pp_done),
        .pp_data             (pp_data),
        .mac_overflow        (mac_overflow),
        .status_idle         (status_idle),
        .status_busy         (status_busy),
        .status_result_valid (status_result_valid),
        .status_acc_ovf_stk  (status_acc_ovf_stk)
    );

    // Command FSM (sec 6)

    cmd_fsm u_fsm (
        .clk          (clk),
        .rst_n        (rst_n_sync),
        .cmd_valid    (cmd_valid),
        .cmd_code     (cmd_code),
        .op_a         (op_a),
        .op_b         (op_b),
        .fsm_state    (fsm_state),
        .mac_overflow (mac_overflow),
        .pp_done      (pp_done),
        .pp_data      (pp_data),
        .soft_reset   (soft_reset),
        .pe_act_in    (pe_act_in),
        .pe_wt_in     (pe_wt_in),
        .pe_acc_in    (pe_acc_in),
        .pe_valid_in  (pe_valid_in),
        .pe_ready_in  (pe_ready_in),
        .pe_acc_out   (pe_acc_out),
        .pe_acc_overflow (pe_acc_overflow),
        .pp_in_data   (pp_in_data),
        .pp_in_valid  (pp_in_valid),
        .pp_in_ready  (pp_in_ready),
        .pp_out_data  (pp_out_data),
        .pp_out_valid (pp_out_valid),
        .pp_out_ready (pp_out_ready)
    );

    // The reused MAC and post-processing pipeline (from lg-npu).
    // Both are reset by core_rst_n so CMD_RESET clears the PE accumulator
    // and the three postproc pipeline registers.

    conv_pe u_pe (
        .clk          (clk),
        .rst_n        (core_rst_n),
        .act_in       (pe_act_in),
        .wt_in        (pe_wt_in),
        .acc_in       (pe_acc_in),
        .acc_out      (pe_acc_out),
        .valid_in     (pe_valid_in),
        .ready_out    (pe_ready_out),
        .valid_out    (pe_valid_out),
        .ready_in     (pe_ready_in),
        .acc_overflow (pe_acc_overflow)
    );

    postproc u_pp (
        .clk         (clk),
        .rst_n       (core_rst_n),
        .bias_val    (bias_val),
        .quant_shift (quant_shift_val),
        .act_mode    (act_mode_bits),
        .in_data     (pp_in_data),
        .in_valid    (pp_in_valid),
        .in_ready    (pp_in_ready),
        .out_data    (pp_out_data),
        .out_valid   (pp_out_valid),
        .out_ready   (pp_out_ready)
    );

    // Heartbeat counter for uo_out[7] (sec 7.2): toggles every 2^19 cycles
    // -> square wave of period 2^20 cycles (~9.5 Hz at the 10 MHz nominal clk).

    reg [19:0] heartbeat_cnt;
    always @(posedge clk or negedge rst_n_sync) begin
        if (!rst_n_sync) heartbeat_cnt <= 20'd0;
        else             heartbeat_cnt <= heartbeat_cnt + 20'd1;
    end

    // Pin outputs (sec 7.2, 7.3)

    assign uo_out = {heartbeat_cnt[19],
                     fsm_state[2:0],
                     status_acc_ovf_stk,
                     status_result_valid,
                     status_busy,
                     status_idle};

    // uio_oe[3] = 1 (MISO drives); all other uio pins are inputs.
    assign uio_oe  = 8'b0000_1000;
    assign uio_out = {4'b0000, spi_miso, 3'b000};

    // Lint tie-offs for unused signals (sec 7.1).
    // ena is fixed high by the TT mux when the project is selected.
    // ui_in is reserved for future use; tied off here.
    // uio_in[7:4] are reserved inputs.
    // pe_valid_out / pe_ready_out are conv_pe's own handshake outputs;
    // the cmd_fsm forces the handshake directly and ignores them.

    wire _unused_ena    = ena;
    wire _unused_ui_in  = &{ui_in,        1'b0};
    wire _unused_uio_in = &{uio_in[7:3],  1'b0};
    wire _unused_pe     = &{pe_valid_out, pe_ready_out, 1'b0};

endmodule
