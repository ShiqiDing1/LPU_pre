import random
import os

# Parameters
MERGE_NUM = 2
DATA_WIDTH = 512
MASK_WIDTH = 64
ADDR_WIDTH = 34
ADDR_STEP = 64
NUM_TEST_CASES = 50

# Output files
INPUT_FILE = "input_data.bin"
GOLDEN_FILE = "golden_data.bin"

def generate_data():
    input_data_list = []
    golden_data_list = []
    
    # Internal state simulation
    req_cntr = 0
    buffer_dat = [0] * MERGE_NUM
    buffer_msk = [0] * MERGE_NUM
    current_batch_start_addr = 0
    prev_addr = 0
    has_data = False
    
    # Helper to flush buffer
    def flush_buffer(reason_last=False):
        nonlocal req_cntr, has_data, current_batch_start_addr, buffer_dat, buffer_msk
        
        # Determine how many items are valid to output
        # In the RTL logic:
        # If we flush due to discontinuity or last, we output what's in the buffer valid up to req_cntr_reg.
        # However, the model needs to be precise. 
        # When flushing, we output a MERGE_NUM width data.
        
        # Prepare valid output line
        out_dat_hex = ""
        out_msk_hex = ""
        
        # RTL Logic simulation:
        # "internal_out_dat[i] = dat_reg[i]; if (i <= req_cntr_reg)"
        # Note: req_cntr in RTL tracks the index for the NEXT incoming item during accumulation, 
        # or the current index being written.
        # Let's trace carefully:
        # req_cntr starts at 0.
        # 1. Receive Item 0: dat_reg[0]=data. req_cntr becomes 1.
        # 2. Receive Item 1: dat_reg[1]=data. req_cntr becomes 0 (width is log2(2)=1 bit? No, logic is logic [$clog2(MERGE_NUM)-1:0] which for 2 is [0:0] i.e. 1 bit).
        # Wait, $clog2(2) = 1. req_cntr is 1 bit: 0, 1.
        # If MERGE_NUM=2, req_cntr counts 0, 1.
        # sign_full = (req_cntr == 1).
        
        # When full (continuous):
        # We accept item 1. req_cntr is 1. sign_full is true.
        # addr_continuous_send_vld goes high NEXT cycle (registered).
        # dat_reg[1] is written.
        # At next cycle, we output.
        
        # For the golden model, we just take the valid items in the buffer and format them.
        # We need to output exactly MERGE_NUM items width, padding with 0s if not full (though invalid).
        
        current_out_dat = 0
        current_out_msk = 0
        
        # Construct the wide output signal
        # Logic: out_dat = {dat_reg[1], dat_reg[0]} because index 1 is high bits?
        # RTL: "out_dat = internal_out_dat", "logic [MERGE_NUM-1:0][DATA_WIDTH-1:0] internal_out_dat"
        # Since it's a packed array, index [MERGE_NUM-1] is the MSB side.
        # So out_dat = {dat_reg[MERGE_NUM-1], ..., dat_reg[0]}
        
        combined_dat = 0
        combined_msk = 0
        
        for i in range(MERGE_NUM):
            # If i < count of items in buffer (which is req_cntr if not reset yet, or MERGE_NUM if full)
            # Actually, let's just use the buffer_dat list we maintained
            d = buffer_dat[i]
            m = buffer_msk[i]
            
            # Shift into position
            combined_dat |= (d << (i * DATA_WIDTH))
            combined_msk |= (m << (i * MASK_WIDTH))
            
        # Format golden output
        # Binary Format: addr(8B), last(1B), data(128B), mask(16B)
        # Total: 153 Bytes
        
        b_addr = current_batch_start_addr.to_bytes(8, 'big')
        b_last = (1 if reason_last else 0).to_bytes(1, 'big')
        b_data = combined_dat.to_bytes(MERGE_NUM*DATA_WIDTH // 8, 'big')
        b_mask = combined_msk.to_bytes(MERGE_NUM*MASK_WIDTH // 8, 'big')
        
        golden_data_list.append(b_addr + b_last + b_data + b_mask)
        
        # Reset state
        req_cntr = 0
        buffer_dat = [0] * MERGE_NUM
        buffer_msk = [0] * MERGE_NUM
        has_data = False
        
    
    # Start generation
    addr = 0x1000
    
    for _ in range(NUM_TEST_CASES):
        # Generate random input
        is_continuous = (random.random() > 0.3) # 70% chance continuous
        is_last = (random.random() > 0.9) # 10% chance last
        
        if is_continuous and has_data:
            addr = prev_addr + ADDR_STEP
        else:
            # Random jump
            addr = random.randint(0, (2**ADDR_WIDTH - 1) // ADDR_STEP) * ADDR_STEP
            
            # If we were tracking a batch, a discontinuity forces a flush of the PREVIOUS batch
            if has_data:
                flush_buffer(reason_last=False)
                # But wait, in RTL, if we send a discontinuous item, it drives "addr_incontinuous_send_vld".
                # This causes the PREVIOUS batch to be sent out in the output logic.
                # AND capture the NEW item into index 0.
        
        data = random.getrandbits(DATA_WIDTH)
        mask = (1 << MASK_WIDTH) - 1 # All valid for simplicity
        
        # Record input
        # Binary Format: valid=1(1B), addr(8B), data(64B), mask(8B), last(1B)
        # Total: 82 Bytes
        
        b_val = (1).to_bytes(1, 'big')
        b_addr = addr.to_bytes(8, 'big')
        b_data = data.to_bytes(DATA_WIDTH // 8, 'big')
        b_mask = mask.to_bytes(MASK_WIDTH // 8, 'big')
        b_lst = (1 if is_last else 0).to_bytes(1, 'big')
        
        input_data_list.append(b_val + b_addr + b_data + b_mask + b_lst)
        
        # Model Logic
        if not has_data:
            # First item in batch
            current_batch_start_addr = addr
            buffer_dat[0] = data
            buffer_msk[0] = mask
            req_cntr = 1
            has_data = True
        else:
            # Buffer has data, add next
            buffer_dat[req_cntr] = data
            buffer_msk[req_cntr] = mask
            req_cntr += 1
            
        prev_addr = addr
        
        # Check conditions to flush
        # 1. Full?
        if req_cntr == MERGE_NUM:
            # Full!
            # If not LAST, we flush because full. 
            # If LAST, we flush because last (priority doesn't matter much for output content here, just trigger)
            flush_buffer(reason_last=is_last) 
            # Note: flush_buffer resets has_data=False.
            # If is_last was true, we are done with this batch.
            # If is_last was false, we are done with this batch (full).
        elif is_last:
            # Not full, but forced flush due to last
            flush_buffer(reason_last=True)
            
    # Write files
    with open(INPUT_FILE, 'wb') as f:
        for item in input_data_list:
            f.write(item)
        
    with open(GOLDEN_FILE, 'wb') as f:
        for item in golden_data_list:
            f.write(item)
        
    print(f"Generated {len(input_data_list)} input vectors and {len(golden_data_list)} golden assertions.")

if __name__ == "__main__":
    generate_data()
