module mac_body #(
    parameter int unsigned A_BIT = 8,
    parameter int unsigned W_BIT = 8,
    parameter int unsigned B_BIT = 32
) (
    input  logic        [A_BIT-1:0] x,
    input  logic signed [W_BIT-1:0] w,
    input  logic signed [B_BIT-1:0] cas_in,
    output logic signed [B_BIT-1:0] cas_out
);
    logic signed [B_BIT-1:0] prod;
    assign prod    = x * w; 
    assign cas_out = prod + cas_in;
endmodule

module mac_tail #(
    parameter int unsigned A_BIT = 8,
    parameter int unsigned W_BIT = 8,
    parameter int unsigned B_BIT = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    en,
    input  logic                    dat_vld,
    input  logic                    clr,
    input  logic        [A_BIT-1:0] x,
    input  logic signed [W_BIT-1:0] w,
    input  logic signed [B_BIT-1:0] cas_in,
    output logic signed [B_BIT-1:0] acc
);
    logic signed [B_BIT-1:0] prod;
    logic signed [B_BIT-1:0] sum;
    logic signed [B_BIT-1:0] acc_r;
    assign prod = x * w;
    assign sum  = prod + cas_in;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_r <= '0;
        end
        else if (en && dat_vld) begin
            if (clr)
                acc_r <= sum;
            else
                acc_r <= acc_r + sum;
        end
    end

    assign acc = acc_r;
endmodule

module conv_mac_array #(
    parameter int unsigned P_ICH = 4,
    parameter int unsigned A_BIT = 8,
    parameter int unsigned W_BIT = 8,
    parameter int unsigned B_BIT = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    en,
    input  logic                    dat_vld,
    input  logic                    clr,
    input  logic        [A_BIT-1:0] x_vec  [P_ICH],
    input  logic signed [W_BIT-1:0] w_vec  [P_ICH],
    output logic signed [B_BIT-1:0] acc
);
    
    logic signed [B_BIT-1:0] mac_cascade [P_ICH];

logic [A_BIT-1:0]        x_aligned [P_ICH];
logic signed [W_BIT-1:0] w_aligned [P_ICH];

logic [P_ICH-1:0] valid_delay;
logic [P_ICH-1:0] clr_delay;

assign mac_cascade[0] = '0;
assign valid_delay[0] = dat_vld;
assign clr_delay[0]   = clr;

generate
    
    for (genvar i = 0; i < P_ICH; i++) begin : gen_align
        logic [A_BIT-1:0]        x_delay [0:i];
        logic signed [W_BIT-1:0] w_delay [0:i];

        assign x_delay[0] = x_vec[i];
        assign w_delay[0] = w_vec[i];

        for (genvar d = 0; d < i; d++) begin : gen_delay
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    x_delay[d+1] <= '0;
                    w_delay[d+1] <= '0;
                end else if (en) begin
                    x_delay[d+1] <= x_delay[d];
                    w_delay[d+1] <= w_delay[d];
                end
            end
        end

        assign x_aligned[i] = x_delay[i];
        assign w_aligned[i] = w_delay[i];
    end

    
    for (genvar i = 0; i < P_ICH-1; i++) begin : gen_mac_body
        logic signed [B_BIT-1:0] body_result;

        mac_body #(
            .A_BIT(A_BIT),
            .W_BIT(W_BIT),
            .B_BIT(B_BIT)
        ) u_mac_body (
            .x      (x_aligned[i]),
            .w      (w_aligned[i]),
            .cas_in (mac_cascade[i]),
            .cas_out(body_result)
        );

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                mac_cascade[i+1] <= '0;
                valid_delay[i+1] <= 1'b0;
                clr_delay[i+1]   <= 1'b0;
            end else if (en) begin
                valid_delay[i+1] <= valid_delay[i];
                clr_delay[i+1]   <= clr_delay[i];

                if (valid_delay[i])
                    mac_cascade[i+1] <= body_result;
            end
        end
    end
endgenerate

mac_tail #(
    .A_BIT(A_BIT),
    .W_BIT(W_BIT),
    .B_BIT(B_BIT)
) u_mac_tail (
    .clk    (clk),
    .rst_n  (rst_n),
    .en     (en),
    .dat_vld(valid_delay[P_ICH-1]),
    .clr    (clr_delay[P_ICH-1]),
    .x      (x_aligned[P_ICH-1]),
    .w      (w_aligned[P_ICH-1]),
    .cas_in (mac_cascade[P_ICH-1]),
    .acc    (acc)
);
endmodule