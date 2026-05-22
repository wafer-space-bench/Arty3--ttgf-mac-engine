// ============================================================================
// pipeline_reg.sv - Single pipeline register stage with valid/ready
// ============================================================================

module pipeline_reg #(
    parameter int DATA_W = 32
) (
    input  logic                clk,
    input  logic                rst_n,

    // Upstream
    input  logic [DATA_W-1:0]   in_data,
    input  logic                in_valid,
    output logic                in_ready,

    // Downstream
    output logic [DATA_W-1:0]   out_data,
    output logic                out_valid,
    input  logic                out_ready
);

    logic [DATA_W-1:0]  data_r;
    logic               valid_r;
    logic               transfer_in, transfer_out;

    assign transfer_in  = in_valid  & in_ready;
    assign transfer_out = valid_r   & out_ready;
    assign in_ready     = ~valid_r | out_ready;
    assign out_data     = data_r;
    assign out_valid    = valid_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_r <= 1'b0;
            data_r  <= '0;
        end else begin
            if (transfer_in) begin
                data_r  <= in_data;
                valid_r <= 1'b1;
            end else if (transfer_out) begin
                valid_r <= 1'b0;
            end
        end
    end

endmodule : pipeline_reg
