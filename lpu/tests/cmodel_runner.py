#!/usr/bin/env python3
"""Small process-isolated ctypes runner for the latest cmodel C API."""

import argparse
import ctypes
import os
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "lib" / "liblpu_func.so"


def find_library():
    if LIB_PATH.is_file():
        return LIB_PATH
    raise FileNotFoundError(f"tutorial cmodel library not found: {LIB_PATH}")


def make_ordered_replay(taskbin, output_dir):
    """Add the replay loader's fourth header word to a legacy 12-byte taskbin."""
    raw = taskbin.read_bytes()
    if len(raw) < 12:
        raise ValueError("taskbin header is truncated")
    instr_size, input_size, output_size = struct.unpack_from("<III", raw)
    expected = 12 + instr_size + input_size + output_size
    if len(raw) != expected:
        raise ValueError(f"taskbin size mismatch: expected {expected}, got {len(raw)}")
    replay = output_dir / "ordered_replay.bin"
    replay.write_bytes(
        struct.pack("<IIII", instr_size, input_size, output_size, instr_size + 4)
        + raw[12:]
    )
    return replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("taskbin", type=Path)
    parser.add_argument("mode", choices=("sequential", "parallel"))
    parser.add_argument("gm_addr", type=int)
    parser.add_argument("size", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lib = ctypes.CDLL(str(find_library()))
    lib.lpu_func_create.restype = ctypes.c_void_p
    lib.lpu_func_destroy.argtypes = [ctypes.c_void_p]
    lib.lpu_func_load_task_bin.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lpu_func_load_task_bin.restype = ctypes.c_int
    lib.lpu_func_load_task_replay_bin.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lpu_func_load_task_replay_bin.restype = ctypes.c_int
    lib.lpu_func_execute.argtypes = [ctypes.c_void_p]
    lib.lpu_func_execute.restype = ctypes.c_int
    lib.lpu_func_execute_sequential.argtypes = [ctypes.c_void_p]
    lib.lpu_func_execute_sequential.restype = ctypes.c_int
    lib.lpu_func_read_buffer.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
    ]
    lib.lpu_func_read_buffer.restype = ctypes.c_int

    handle = lib.lpu_func_create()
    if not handle:
        raise RuntimeError("lpu_func_create failed")
    try:
        load_path = args.taskbin
        if args.mode == "sequential":
            load_path = make_ordered_replay(args.taskbin, args.output.parent)
        loader = (
            lib.lpu_func_load_task_replay_bin
            if args.mode == "sequential"
            else lib.lpu_func_load_task_bin
        )
        rc = loader(handle, os.fsencode(load_path))
        if rc != 0:
            raise RuntimeError(f"load_task_bin failed: {rc}")
        execute = (
            lib.lpu_func_execute_sequential
            if args.mode == "sequential"
            else lib.lpu_func_execute
        )
        rc = execute(handle)
        if rc != 0:
            raise RuntimeError(f"execute failed: {rc}")
        out = (ctypes.c_uint8 * args.size)()
        rc = lib.lpu_func_read_buffer(handle, 0, args.gm_addr, out, args.size)
        if rc != args.size:
            raise RuntimeError(f"read GM failed: expected {args.size}, got {rc}")
        args.output.write_bytes(bytes(out))
    finally:
        lib.lpu_func_destroy(handle)


if __name__ == "__main__":
    main()
