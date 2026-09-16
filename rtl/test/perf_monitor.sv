`timescale 1ns / 1ps

module perf_monitor #(
    parameter longint unsigned THEORETICAL_CYCLES = 1
) (
    input logic clk,
    input logic rst_n,
    input logic input_fire,
    input logic output_fire
);

    int cycle_count;
    int input_transfer_count;
    int output_transfer_count;
    int first_input_cycle;
    int last_input_cycle;
    int first_output_cycle;
    int last_output_cycle;
    int timeout_cycles;

    initial begin
        timeout_cycles = 10000000;
        void'($value$plusargs("PERF_TIMEOUT_CYCLES=%d", timeout_cycles));
        assert (THEORETICAL_CYCLES > 0)
        else $fatal(1, "THEORETICAL_CYCLES must be greater than zero");
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count           <= 0;
            input_transfer_count  <= 0;
            output_transfer_count <= 0;
            first_input_cycle     <= -1;
            last_input_cycle      <= -1;
            first_output_cycle    <= -1;
            last_output_cycle     <= -1;
        end else begin
            cycle_count <= cycle_count + 1;

            if (input_fire) begin
                if (first_input_cycle < 0) begin
                    first_input_cycle <= cycle_count;
                end
                last_input_cycle     <= cycle_count;
                input_transfer_count <= input_transfer_count + 1;
            end

            if (output_fire) begin
                if (first_output_cycle < 0) begin
                    first_output_cycle <= cycle_count;
                end
                last_output_cycle     <= cycle_count;
                output_transfer_count <= output_transfer_count + 1;
            end

            if (cycle_count >= timeout_cycles) begin
                report();
                $fatal(1, "PERF TIMEOUT after %0d cycles", timeout_cycles);
            end
        end
    end

    task automatic report;
        int input_window_cycles;
        int output_window_cycles;
        int measurement_cycles;
        real input_throughput;
        real output_throughput;
        real performance_efficiency;
        begin
            input_window_cycles    = 0;
            output_window_cycles   = 0;
            measurement_cycles     = 0;
            input_throughput        = 0.0;
            output_throughput       = 0.0;
            performance_efficiency = 0.0;

            if (input_transfer_count > 0) begin
                input_window_cycles = last_input_cycle - first_input_cycle + 1;
                input_throughput = $itor(input_transfer_count) / $itor(input_window_cycles);
            end
            if (output_transfer_count > 0) begin
                output_window_cycles = last_output_cycle - first_output_cycle + 1;
                output_throughput = $itor(output_transfer_count) / $itor(output_window_cycles);
            end

            $display("\n========================================");
            $display("Performance Summary (saturated I/O)");
            $display("========================================");
            $display("Input transfers:                %0d", input_transfer_count);
            $display("Output transfers:               %0d\n", output_transfer_count);
            $display("Input throughput:               %.6f transfers/cycle", input_throughput);
            $display("Steady output throughput:       %.6f transfers/cycle\n", output_throughput);

            if ((input_transfer_count == 0) || (output_transfer_count == 0) ||
                (last_output_cycle < first_input_cycle)) begin
                $display("PERF ERROR: no complete input/output traffic was observed");
                $display("Startup latency:                N/A");
                $display("Drain latency:                  N/A");
                $display("Theoretical cycles:             %0d cycles", THEORETICAL_CYCLES);
                $display("Actual cycles:                  N/A");
                $display("Performance efficiency:         N/A");
            end else begin
                measurement_cycles = last_output_cycle - first_input_cycle + 1;
                performance_efficiency =
                    $itor(THEORETICAL_CYCLES) / $itor(measurement_cycles);

                $display("Startup latency:                %0d cycles", first_output_cycle - first_input_cycle);
                $display("Drain latency:                  %0d cycles\n", last_output_cycle - last_input_cycle);
                $display("Theoretical cycles:             %0d cycles", THEORETICAL_CYCLES);
                $display("Actual cycles:                  %0d cycles", measurement_cycles);
                $display("Performance efficiency:         %.4f%%", performance_efficiency * 100.0);
            end

            $display("========================================\n");
        end
    endtask

endmodule
