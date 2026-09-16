module mac_array #(
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

    logic signed [B_BIT-1:0] mac_cascade[P_ICH+1];

    assign mac_cascade[0] = '0;

    logic signed [A_BIT:0]        x_aligned [P_ICH];
    logic signed [W_BIT-1:0] w_aligned [P_ICH];
    logic [P_ICH-1:0] valid_delay;
    logic [P_ICH-1:0] clr_delay;

    assign valid_delay[0] = dat_vld;
    assign clr_delay[0]   = clr;

    generate
        for (genvar i = 0; i < P_ICH; i++) begin : gen_align
            logic signed [B_BIT-1:0] acc_r;
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
        for (genvar i = 0; i < P_ICH-1; i++) begin : gen_mac
        logic signed [B_BIT-1:0] acc_r;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                acc_r           <= '0;
                clr_delay[i+1]   <= 1'b0;
                valid_delay[i+1] <= 1'b0;
            end else if (en) begin
                clr_delay[i+1]   <= clr_delay[i];
                valid_delay[i+1] <= valid_delay[i];

                if (valid_delay[i]) begin
                    acc_r <=x_aligned[i] * w_aligned[i] + mac_cascade[i];
                end
            end
        end

        assign mac_cascade[i+1] = acc_r;
    end
endgenerate


logic signed [B_BIT-1:0] tail_acc_r;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        tail_acc_r <= '0;
    end else if (en && valid_delay[P_ICH-1]) begin
        if (clr_delay[P_ICH-1]) begin
            tail_acc_r <= ($signed(x_aligned[P_ICH-1]) * w_aligned[P_ICH-1])+ mac_cascade[P_ICH-1];
        end else begin
            tail_acc_r <= tail_acc_r + ($signed(x_aligned[P_ICH-1]) * w_aligned[P_ICH-1]) + mac_cascade[P_ICH-1];
        end
    end
end

assign mac_cascade[P_ICH] = tail_acc_r;

    assign acc = mac_cascade[P_ICH];

endmodule
