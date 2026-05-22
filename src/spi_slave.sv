// ============================================================================
// spi_slave.sv - SPI Mode 0 slave, MSB-first, 2-byte framed transactions
//
// See docs/SPEC.md sections 3 (SPI protocol) and 9.1 (clock-domain crossing).
//
// Frame layout (16 SCLK cycles per CS_N-framed transaction):
//   Byte 0 (host -> chip): HEADER
//     bit 7    : R/W#  (1 = write, 0 = read)
//     bits 6:0 : 7-bit register address
//   Byte 1:
//     Write : host -> chip data byte
//     Read  : chip -> host data byte (MOSI don't-care)
//
// Snapshot edge for reads: falling edge of the 8th SCLK of byte 0, i.e. the
// edge immediately after the LSB of the address (header bit 0) is sampled.
//
// MISO is actively driven at all times. Held at 1'b0 whenever CS_N is high
// (the pad is not tri-stated; uio_oe[3] is hard-wired to 1 at the top).
// ============================================================================

`default_nettype none

module spi_slave (
    input  logic       clk,
    input  logic       rst_n,

    // Raw pad signals (asynchronous to clk)
    input  logic       spi_cs_n_pad,
    input  logic       spi_sclk_pad,
    input  logic       spi_mosi_pad,
    output logic       spi_miso,

    // Regfile port (system clock domain)
    output logic [6:0] rf_addr,
    output logic [7:0] rf_wdata,
    output logic       rf_we,    // 1-cycle pulse on a successful write commit
    output logic       rf_re,    // 1-cycle pulse on the read-snapshot edge
    input  logic [7:0] rf_rdata
);
    // 2-flop synchronisers + one-cycle delays for edge detection (sec 9.1)

    logic [1:0] cs_n_ff;
    logic [1:0] sclk_ff;
    logic [1:0] mosi_ff;
    logic       cs_n_sync_d;
    logic       sclk_sync_d;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cs_n_ff     <= 2'b11;   // idle-high
            sclk_ff     <= 2'b00;
            mosi_ff     <= 2'b00;
            cs_n_sync_d <= 1'b1;
            sclk_sync_d <= 1'b0;
        end else begin
            cs_n_ff     <= {cs_n_ff[0], spi_cs_n_pad};
            sclk_ff     <= {sclk_ff[0], spi_sclk_pad};
            mosi_ff     <= {mosi_ff[0], spi_mosi_pad};
            cs_n_sync_d <=  cs_n_ff[1];
            sclk_sync_d <=  sclk_ff[1];
        end
    end

    wire cs_n_sync = cs_n_ff[1];
    wire sclk_sync = sclk_ff[1];
    wire mosi_sync = mosi_ff[1];

    wire cs_n_rise =  cs_n_sync   &  ~cs_n_sync_d;
    wire cs_n_fall = ~cs_n_sync   &   cs_n_sync_d;
    wire sclk_rise = (~cs_n_sync) & ( sclk_sync & ~sclk_sync_d);
    wire sclk_fall = (~cs_n_sync) & (~sclk_sync &  sclk_sync_d);

    // Shift register, bit counter, header capture
    //
    // bit_cnt counts how many SCLK rising edges have occurred since the most
    // recent CS_N falling edge. After 8 rising edges the header byte has been
    // shifted into rx_shift; after 16 rising edges the data byte (or the
    // ignored MOSI of a read) has been shifted in.

    logic [4:0] bit_cnt;     // 0 .. 16
    logic [7:0] rx_shift;
    logic [6:0] addr_r;
    logic       rw_r;        // captured header bit 7 (1 = write, 0 = read)
    logic [7:0] tx_shift;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_cnt  <= 5'd0;
            rx_shift <= 8'd0;
            addr_r   <= 7'd0;
            rw_r     <= 1'b0;
            tx_shift <= 8'd0;
        end else begin
            // Start-of-frame: reset counter and shift register
            if (cs_n_fall) begin
                bit_cnt  <= 5'd0;
                rx_shift <= 8'd0;
            end else if (sclk_rise) begin
                rx_shift <= {rx_shift[6:0], mosi_sync};
                if (bit_cnt != 5'd16) begin
                    bit_cnt <= bit_cnt + 5'd1;
                end
                // 8th rising edge: header byte fully shifted in.
                // The new (post-shift) value is {rx_shift[6:0], mosi_sync};
                // bit 7 -> rw, bits 6:0 -> address.
                if (bit_cnt == 5'd7) begin
                    rw_r   <= rx_shift[6];
                    addr_r <= {rx_shift[5:0], mosi_sync};
                end
            end

            // MISO shift register.
            //   - At sclk_fall with bit_cnt == 8 and rw == 0 (read): load the
            //     read snapshot from rf_rdata.
            //   - At later sclk_fall edges in byte 1: shift left so MSB-first
            //     output continues.
            if (sclk_fall) begin
                if (bit_cnt == 5'd8 && !rw_r) begin
                    tx_shift <= rf_rdata;
                end else if (bit_cnt > 5'd8 && bit_cnt <= 5'd15) begin
                    tx_shift <= {tx_shift[6:0], 1'b0};
                end
            end
        end
    end

    // Outputs

    // MISO: driven actively at all times. Held at 0 when CS_N high (sec 7.3).
    assign spi_miso = (~cs_n_sync) ? tx_shift[7] : 1'b0;

    // Regfile interface
    assign rf_addr  = addr_r;
    assign rf_wdata = rx_shift;

    // Write commit fires once, on the synchronised CS_N rising edge of a
    // complete 16-bit write transaction. Aborted transactions (CS_N rises
    // with bit_cnt < 16) do not commit.
    assign rf_we    = cs_n_rise & rw_r & (bit_cnt == 5'd16);

    // Read snapshot pulse: the first sclk_fall after the 8th sclk_rise of
    // a read transaction (i.e. immediately after the address LSB lands).
    assign rf_re    = sclk_fall & (bit_cnt == 5'd8) & ~rw_r;

endmodule : spi_slave
