// ============================================================================
// cmd_fsm.sv - Command sequencer for the SPI-controlled MAC engine
//
// See docs/SPEC.md section 6.
//
// States (3-bit encoding; code 3'b111 is unreachable):
//   ST_IDLE    = 3'b000  Wait for cmd_valid
//   ST_MAC     = 3'b001  1-cycle MAC: acc <- acc + op_a * op_b
//   ST_CLR     = 3'b010  1-cycle accumulator clear via zero-injection
//   ST_PP_FEED = 3'b011  Drive the accumulator into postproc.in
//   ST_PP_WAIT = 3'b100  Wait for postproc.out_valid, latch RESULT
//   ST_SOFTRST = 3'b101  Pulse soft_reset for one cycle
//   ST_DOT4    = 3'b110  4-cycle burst MAC over op_{a,b}{0..3}
//
// State transitions are registered. next_state is combinational on the
// current state and cmd_valid. A same-cycle return to ST_IDLE accepts a
// new cmd_valid that lands on that cycle (sec 4.2).
// ============================================================================

`default_nettype none

module cmd_fsm #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
) (
    input  logic                       clk,
    input  logic                       rst_n,

    // From regfile
    input  logic                       cmd_valid,
    input  logic [7:0]                 cmd_code,
    input  logic signed [DATA_W-1:0]   op_a,
    input  logic signed [DATA_W-1:0]   op_b,
    // Lane 1..3 operands for CMD_DOT4. Lane 0 reuses op_a / op_b.
    input  logic signed [DATA_W-1:0]   op_a1,
    input  logic signed [DATA_W-1:0]   op_b1,
    input  logic signed [DATA_W-1:0]   op_a2,
    input  logic signed [DATA_W-1:0]   op_b2,
    input  logic signed [DATA_W-1:0]   op_a3,
    input  logic signed [DATA_W-1:0]   op_b3,

    // To regfile
    output logic [2:0]                 fsm_state,
    output logic                       mac_overflow,
    output logic                       pp_done,
    output logic signed [DATA_W-1:0]   pp_data,
    output logic                       soft_reset,

    // conv_pe handshake / operands
    output logic signed [DATA_W-1:0]   pe_act_in,
    output logic signed [DATA_W-1:0]   pe_wt_in,
    output logic signed [ACC_W-1:0]    pe_acc_in,
    output logic                       pe_valid_in,
    output logic                       pe_ready_in,
    input  logic signed [ACC_W-1:0]    pe_acc_out,
    input  logic                       pe_acc_overflow,

    // postproc handshake
    output logic signed [ACC_W-1:0]    pp_in_data,
    output logic                       pp_in_valid,
    input  logic                       pp_in_ready,
    input  logic signed [DATA_W-1:0]   pp_out_data,
    input  logic                       pp_out_valid,
    output logic                       pp_out_ready
);

    // State encoding

    typedef enum logic [2:0]
	{
        ST_IDLE    = 3'b000,
        ST_MAC     = 3'b001,
        ST_CLR     = 3'b010,
        ST_PP_FEED = 3'b011,
        ST_PP_WAIT = 3'b100,
        ST_SOFTRST = 3'b101,
        ST_DOT4    = 3'b110
    }   state_e;

    state_e state_r;
    state_e next_state;

    // 2-bit lane index for the burst MAC. Held at zero outside ST_DOT4,
    // increments each cycle inside ST_DOT4.
    logic [1:0] lane_cnt_r;

    // Lane-selected operands for the conv_pe datapath. The mux is driven
    // by lane_cnt_r in ST_DOT4 and falls back to op_a / op_b otherwise so
    // ST_MAC reuses the same Mealy outputs.
    logic signed [DATA_W-1:0] lane_a;
    logic signed [DATA_W-1:0] lane_b;

    always_comb begin
        unique case (lane_cnt_r)
            2'd0: begin lane_a = op_a;  lane_b = op_b;  end
            2'd1: begin lane_a = op_a1; lane_b = op_b1; end
            2'd2: begin lane_a = op_a2; lane_b = op_b2; end
            2'd3: begin lane_a = op_a3; lane_b = op_b3; end
            default: begin lane_a = '0; lane_b = '0; end
        endcase
    end

    // Command opcodes (mirrors sec 4.2). NOP (8'h00) and unknown opcodes
    // resolve to ST_IDLE via decode_cmd's default branch.

    localparam logic [7:0] CMD_MAC      = 8'h01;
    localparam logic [7:0] CMD_CLR_ACC  = 8'h02;
    localparam logic [7:0] CMD_POSTPROC = 8'h03;
    localparam logic [7:0] CMD_DOT4     = 8'h04;
    localparam logic [7:0] CMD_RESET    = 8'hFF;

    // Decode an opcode into the state to enter on the next clock edge.
    // NOP and unknown opcodes resolve to ST_IDLE (sec 4.2, 6).
    function automatic state_e decode_cmd(input logic [7:0] code);
        unique case (code)
            CMD_MAC:      decode_cmd = ST_MAC;
            CMD_CLR_ACC:  decode_cmd = ST_CLR;
            CMD_POSTPROC: decode_cmd = ST_PP_FEED;
            CMD_DOT4:     decode_cmd = ST_DOT4;
            CMD_RESET:    decode_cmd = ST_SOFTRST;
            default:      decode_cmd = ST_IDLE;
        endcase
    endfunction

    // Same-cycle return to IDLE counts as IDLE (sec 4.2): any state that
    // terminates this cycle (ST_MAC, ST_CLR, ST_SOFTRST, and ST_PP_WAIT
    // when pp_out_valid asserts) re-dispatches on cmd_valid in this same
    // combinational pass so a host write that lands on the completion
    // cycle is accepted instead of dropped.

    // Next-state logic

    always_comb begin
        state_e idle_next;
        idle_next = ST_IDLE;
        if (cmd_valid) idle_next = decode_cmd(cmd_code);
        next_state = state_r;
        unique case (state_r)
            ST_IDLE:    next_state = idle_next;
            ST_MAC:     next_state = idle_next;
            ST_CLR:     next_state = idle_next;
            ST_PP_FEED: if (pp_in_ready)  next_state = ST_PP_WAIT;
            ST_PP_WAIT: if (pp_out_valid) next_state = idle_next;
            ST_SOFTRST: next_state = idle_next;
            ST_DOT4:    if (lane_cnt_r == 2'd3) next_state = idle_next;
            default:    next_state = ST_IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state_r <= ST_IDLE;
        else        state_r <= next_state;
    end

    // Lane counter: held at zero outside ST_DOT4; increments each cycle
    // while in ST_DOT4. Wraps on the cycle where it reaches 3 because the
    // FSM transitions out on that same cycle.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                     lane_cnt_r <= 2'd0;
        else if (state_r == ST_DOT4)    lane_cnt_r <= lane_cnt_r + 2'd1;
        else                            lane_cnt_r <= 2'd0;
    end

    assign fsm_state = state_r;

    // MAC overflow flag is sourced directly from conv_pe (sec 5.2). conv_pe
    // already computes the sign-extended product and the 32-bit sum, so we
    // reuse its combinational overflow output rather than duplicating the
    // multiplier and adder here.

    // Datapath outputs (Mealy on state)

    always_comb begin
        pe_act_in    = '0;
        pe_wt_in     = '0;
        pe_acc_in    = '0;
        pe_valid_in  = 1'b0;
        pe_ready_in  = 1'b0;
        pp_in_data   = '0;
        pp_in_valid  = 1'b0;
        pp_out_ready = 1'b0;
        mac_overflow = 1'b0;
        soft_reset   = 1'b0;

        unique case (state_r)
            ST_IDLE: ; // all defaults

            ST_MAC: begin
                // Force do_accept = 1 inside conv_pe by asserting both
                // valid_in and ready_in. ready_out = ~valid_r | ready_in = 1
                // regardless of the PE's stuck valid_r (sec 2.2 / 6.1).
                pe_act_in    = op_a;
                pe_wt_in     = op_b;
                pe_acc_in    = pe_acc_out;
                pe_valid_in  = 1'b1;
                pe_ready_in  = 1'b1;
                mac_overflow = pe_acc_overflow;
            end

            ST_DOT4: begin
                // 4 consecutive MACs over (op_a{0..3}, op_b{0..3}), each
                // accumulating into pe_acc_out as in ST_MAC. The lane
                // counter drives the operand mux above; the FSM exits to
                // idle_next on the lane==3 cycle (sec 6.x).
                pe_act_in    = lane_a;
                pe_wt_in     = lane_b;
                pe_acc_in    = pe_acc_out;
                pe_valid_in  = 1'b1;
                pe_ready_in  = 1'b1;
                mac_overflow = pe_acc_overflow;
            end

            ST_CLR: begin
                // Zero-injection: act_in = wt_in = 0, acc_in = 0 -> product = 0
                // captured into result_r. conv_pe has no synchronous clear input.
                pe_act_in    = '0;
                pe_wt_in     = '0;
                pe_acc_in    = '0;
                pe_valid_in  = 1'b1;
                pe_ready_in  = 1'b1;
            end

            ST_PP_FEED: begin
                pp_in_data   = pe_acc_out;
                pp_in_valid  = 1'b1;
                pp_out_ready = 1'b1;  // keep the downstream path open
            end

            ST_PP_WAIT: begin
                pp_in_valid  = 1'b0;
                pp_out_ready = 1'b1;
            end

            ST_SOFTRST: begin
                // 1-cycle pulse. The top level ORs this with the hard reset
                // to clear conv_pe.result_r and the postproc pipeline regs;
                // the regfile clears its accumulator-related state directly.
                soft_reset   = 1'b1;
            end

            default: ;
        endcase
    end

    // Post-processing completion (1-cycle pulse + data byte to regfile)
    assign pp_done = (state_r == ST_PP_WAIT) & pp_out_valid;
    assign pp_data = pp_out_data;

endmodule : cmd_fsm
