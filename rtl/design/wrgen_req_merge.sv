module wrgen_req_merge #(
    parameter int MERGE_NUM  = 2,
    parameter int DATA_WIDTH = 512,
    parameter int MASK_WIDTH = 64,
    parameter int ADDR_WIDTH = 34,
    parameter int ADDR_STEP  = 64
) (
    input  logic                                    clk,
    input  logic                                    rst_n,

    // Input Interface
    input  logic                                    in_vld,
    input  logic [ADDR_WIDTH-1:0]                   in_addr,
    input  logic [DATA_WIDTH-1:0]                   in_dat,
    input  logic [MASK_WIDTH-1:0]                   in_msk,
    input  logic                                    in_lst,
    output logic                                    in_rdy,

    // Output Interface
    output logic                                    out_vld,
    output logic [ADDR_WIDTH-1:0]                   out_addr,
    output logic [MERGE_NUM*DATA_WIDTH-1:0]         out_dat,
    output logic [MERGE_NUM*MASK_WIDTH-1:0]         out_msk,
    output logic                                    out_lst,
    input  logic                                    out_rdy
);

    logic [MERGE_NUM-1:0][DATA_WIDTH-1:0] data_buffer;
    logic [MERGE_NUM-1:0][MASK_WIDTH-1:0] mask_buffer;
    logic [ADDR_WIDTH-1:0] last_addr_reg;
    logic [$clog2(MERGE_NUM+1)-1:0] cnt;

    assign is_continuous = (in_addr == last_addr_reg + ADDR_STEP);
    
    assign in_rdy = !out_vld || out_rdy;
    assign in_fire = in_vld & in_rdy;
    assign out_fire = out_vld & out_rdy;

    logic last_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            last_reg <= '0;
        end
        else if(in_fire) begin
            last_reg <= in_lst;
        end
        else if (out_fire)
            last_reg <= 1'b0;
    end

    assign out_lst = last_reg;
    

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            last_addr_reg <= '0;
        end
        else if(in_fire) begin
            last_addr_reg <= in_addr;
        end

    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            cnt <= '0;
        end
        else if (in_fire) begin
            cnt <= out_fire ? 1 : cnt + 1'b1;
        end
        else if (out_fire) begin
            cnt <= '0;
        end
        

    end
    
    assign out_vld = (cnt != 0) && ((cnt == MERGE_NUM) ||last_reg ||(in_vld && !is_continuous));

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            data_buffer <= '0;
            mask_buffer <= '0;
        end
        else if(in_fire) begin

                if (out_fire || cnt == 0) begin
            
            for (int j = 0; j < MERGE_NUM; j++) begin
                if (j == 0) begin
                    data_buffer[j] <= in_dat;
                    mask_buffer[j] <= in_msk;
                end else begin
                    data_buffer[j] <= '0;
                    mask_buffer[j] <= '0;
                end
            end
        end else begin
            data_buffer[cnt] <= in_dat;
            mask_buffer[cnt] <= in_msk;
        end
    end else if (out_fire) begin
        data_buffer <= '0;
        mask_buffer <= '0;
    end
end
    
    assign out_dat = data_buffer;
    assign out_msk = mask_buffer;

    logic [ADDR_WIDTH-1:0] first_addr_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            first_addr_reg <= '0;
        else if (in_fire && (out_fire || cnt == 0))
            first_addr_reg <= in_addr;
    end

    assign out_addr = first_addr_reg;

endmodule
