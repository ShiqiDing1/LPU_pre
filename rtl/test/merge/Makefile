# Makefile for wrgen_req_merge testbench with VCS

# Project paths
PROJECT_ROOT := $(shell cd ../.. && pwd)
SET_ENV_SCRIPT := $(PROJECT_ROOT)/set_env.sh

# Source set_env.sh to get PROJECT_PATH
SHELL := /bin/bash
export PROJECT_PATH := $(shell source $(SET_ENV_SCRIPT) > /dev/null 2>&1 && echo $$PROJECT_PATH)

# Source files
TB_DIR = .
DESIGN_DIR = ../../design
DATA_DIR = ../../data

TB_SRCS = $(TB_DIR)/tb_wrgen_req_merge.sv
DESIGN_SRCS = $(DESIGN_DIR)/wrgen_req_merge.sv
PERF_SRCS = ../perf_monitor.sv

# VCS settings
VCS = vcs
VCS_FLAGS = -full64 \
            -sverilog \
            -timescale=1ns/1ps \
            +v2k \
            -debug_access+all \
            -lca \
            -kdb \
            +lint=TFIPC-L \
            +warn=noUFTMD \
            +incdir+$(PROJECT_PATH)/data \
            -P ${VERDI_HOME}/share/PLI/VCS/LINUX64/novas.tab \
            ${VERDI_HOME}/share/PLI/VCS/LINUX64/pli.a

# Simulation flags for FSDB dumping
FSDB_FILE = tb_wrgen_req_merge.fsdb
SIM_FLAGS = +vcs+flush+all \
            +fsdbfile+$(FSDB_FILE) \
            +PROJECT_PATH=$(PROJECT_PATH)

# Output files
SIMV = simv
PERF_SIMV = simv_perf
LOG_FILE = sim.log
PERF_LOG_FILE = perf.log
PERF_TIMEOUT_CYCLES ?= 10000000

.PHONY: all compile sim func perf clean wave verdi help check_env

all: sim

# Check if PROJECT_PATH is set
check_env:
	$(if $(PROJECT_PATH),, $(error PROJECT_PATH not set! Source set_env.sh first))
	@echo "PROJECT_PATH is set to: $(PROJECT_PATH)"

# Compile with VCS
compile: check_env $(DESIGN_SRCS) $(TB_SRCS)
	@echo "Compiling with VCS..."
	$(VCS) $(VCS_FLAGS) $(DESIGN_SRCS) $(TB_SRCS) -o $(SIMV)
	@echo "Compilation complete!"

# Run simulation
sim: compile
	@echo "Running simulation..."
	./$(SIMV) $(SIM_FLAGS) | tee $(LOG_FILE)
	@echo "Simulation complete!"

# Functional test (same as sim)
func: sim

# Performance test with a saturated source and an always-ready sink
perf: check_env $(DESIGN_SRCS) $(TB_SRCS) $(PERF_SRCS)
	@echo "Compiling for performance test..."
	$(VCS) $(VCS_FLAGS) +define+PERF_TEST $(DESIGN_SRCS) $(PERF_SRCS) $(TB_SRCS) -o $(PERF_SIMV)
	@echo "Running performance simulation..."
	set -o pipefail; ./$(PERF_SIMV) $(SIM_FLAGS) +PERF_TIMEOUT_CYCLES=$(PERF_TIMEOUT_CYCLES) | tee $(PERF_LOG_FILE)
	@echo "Performance simulation complete!"

# View waveform with Verdi
verdi: 
	@if [ ! -f $(FSDB_FILE) ]; then \
		echo "Error: $(FSDB_FILE) not found!"; \
		exit 1; \
	fi
	@echo "Opening Verdi to view waveform..."
	verdi -ssf $(FSDB_FILE) -nologo &

# Clean generated files
clean:
	rm -rf $(SIMV) $(SIMV).daidir $(PERF_SIMV) $(PERF_SIMV).daidir csrc ucli.key
	rm -f $(FSDB_FILE) $(LOG_FILE) $(PERF_LOG_FILE)
	rm -f *.log *.vpd *.fsdb *.vcd
	rm -rf verdiLog novas* *fsdb* *.conf
	rm -rf verdi_config_file

help:
	@echo "Available targets:"
	@echo "  make sim       - Compile and run simulation"
	@echo "  make func      - Functional test (same as sim)"
	@echo "  make perf      - Saturated-I/O latency and throughput test"
	@echo "  make verdi     - Open waveform viewer (Verdi)"
	@echo "  make clean     - Remove generated files"
