import os
import sys
import time
import psutil
import torch
import cv2
import numpy as np

# Add parent directory to path so we can import YOLO components
sys.path.append(os.path.abspath('.'))

from models.experimental import attempt_load
from utils.datasets import letterbox

def get_ram_usage():
    """Returns the current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def get_vram_usage():
    """Returns the current VRAM usage in MB (allocated, reserved)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return allocated, reserved, max_allocated
    return 0.0, 0.0, 0.0

def main():
    # Setup frame skip from arguments or default to 5
    frame_skip = 5
    if len(sys.argv) > 1:
        try:
            frame_skip = int(sys.argv[1])
        except ValueError:
            pass

    print("====================================================")
    print(" YOLO-CROWD Memory & Performance Benchmarking Utility")
    print(f" Mode: Video Decoding + Inference (Frame Skip: {frame_skip})")
    print("====================================================")
    
    # 1. Baseline measurements
    ram_init = get_ram_usage()
    print(f"[*] Initial System RAM Usage: {ram_init:.2f} MB")
    
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f"[*] Targeting Device: {device_name.upper()}")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        vram_alloc, vram_res, _ = get_vram_usage()
        print(f"[*] Initial GPU VRAM Allocated: {vram_alloc:.2f} MB")
        print(f"[*] Initial GPU VRAM Reserved: {vram_res:.2f} MB")
    else:
        print("[*] GPU not available. GPU VRAM tracking will be skipped.")
    
    # 2. Load Model
    print("\n[~] Loading model 'yolo-crowd.pt'...")
    t_load_start = time.time()
    model = attempt_load("yolo-crowd.pt", map_location=device)
    model.eval()
    t_load_end = time.time()
    print(f"[+] Model loaded in {t_load_end - t_load_start:.2f} seconds.")
    
    ram_after_load = get_ram_usage()
    print(f"[*] RAM Usage after Model Load: {ram_after_load:.2f} MB (increased by {ram_after_load - ram_init:.2f} MB)")
    
    # 3. Load Sample Video
    video_path = 'test22.mp4'
    if not os.path.exists(video_path):
        print(f"[-] Video file '{video_path}' not found. Exiting.")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[-] Error opening video stream.")
        sys.exit(1)
        
    # 4. Continuous Inference Loop for 60 Seconds
    benchmark_duration = 60.0  # seconds
    print(f"\n[~] Starting continuous video processing benchmark for {benchmark_duration} seconds...")
    
    ram_history = []
    vram_history = []
    inference_times = []
    
    start_time = time.time()
    end_time = start_time + benchmark_duration
    
    frame_count = 0
    inference_count = 0
    
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            # Loop the video back to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame_count += 1
        
        # Check if we should skip this frame for model inference
        should_run_inference = (frame_skip == 1) or (frame_count % frame_skip == 1)
        
        if should_run_inference:
            t0 = time.time()
            
            # Preprocess frame
            img_padded, _, _ = letterbox(frame, new_shape=640, auto=True, stride=32)
            img_tensor = img_padded.transpose(2, 0, 1)
            img_tensor = np.ascontiguousarray(img_tensor)
            img_tensor = torch.from_numpy(img_tensor).to(device)
            img_tensor = img_tensor.float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            if device_name == 'cuda':
                model.half()
                img_tensor = img_tensor.half()
                
            # Run inference
            with torch.no_grad():
                _ = model(img_tensor)[0]
                
            t1 = time.time()
            inference_times.append(t1 - t0)
            inference_count += 1
            
        # Measure RAM/VRAM
        current_ram = get_ram_usage()
        ram_history.append(current_ram)
        
        if torch.cuda.is_available():
            current_vram, _, _ = get_vram_usage()
            vram_history.append(current_vram)
            
        # Print status every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"    Processed: {frame_count} frames (Inferences: {inference_count}) | Elapsed: {elapsed:.1f}s | Speed: {fps:.1f} FPS | RAM: {current_ram:.1f} MB")
            
    actual_duration = time.time() - start_time
    cap.release()
    
    # 5. Analysis & Summary
    ram_avg = np.mean(ram_history)
    ram_max = np.max(ram_history)
    
    avg_inference_latency = np.mean(inference_times) * 1000 if inference_times else 0.0 # ms
    overall_fps = frame_count / actual_duration
    
    print("\n" + "=" * 52)
    print(" BENCHMARK RESULTS")
    print("=" * 52)
    print(f"Total Duration:         {actual_duration:.2f} seconds")
    print(f"Total Video Frames Read:{frame_count}")
    print(f"Total Inferences Run:   {inference_count}")
    print(f"Overall Video Playback: {overall_fps:.2f} FPS")
    print(f"Avg Inference Latency:  {avg_inference_latency:.2f} ms per active frame")
    print("-" * 52)
    print(" SYSTEM RAM USAGE:")
    print(f"  Baseline RAM:         {ram_init:.2f} MB")
    print(f"  Post-Load RAM:        {ram_after_load:.2f} MB")
    print(f"  Average Running RAM:  {ram_avg:.2f} MB")
    print(f"  Peak Running RAM:     {ram_max:.2f} MB")
    
    if torch.cuda.is_available():
        vram_avg = np.mean(vram_history)
        _, _, max_allocated = get_vram_usage()
        print("-" * 52)
        print(" GPU VRAM USAGE:")
        print(f"  Average Running VRAM: {vram_avg:.2f} MB")
        print(f"  Peak Allocated VRAM:  {max_allocated:.2f} MB")
    print("====================================================")

if __name__ == "__main__":
    main()
