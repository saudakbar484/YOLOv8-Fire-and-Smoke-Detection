import os
import sys
import time
import torch
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("      YOLOv8 Fire and Smoke Detection & Tracking")
    print("=" * 60)

    # 1. Device Info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Torch version: {torch.__version__}")
    print(f"[*] Using Device: {device.upper()}")
    if device == "cuda":
        print(f"    - GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"    - CUDA Capability: {torch.cuda.get_device_capability(0)}")

    # 2. Check model weights path
    weights_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
    if not os.path.exists(weights_path):
        print(f"[!] Error: Weights file not found at {weights_path}")
        print("    Please check if the model has been trained successfully.")
        sys.exit(1)

    print(f"[*] Loading YOLOv8 model from: {weights_path}")
    model = YOLO(weights_path)
    
    # Print model details
    names = model.names
    print(f"[*] Model Class Names: {names}")
    
    # 3. Check demo source
    source_video = "demo.mp4"
    if not os.path.exists(source_video):
        print(f"[!] Error: Source video '{source_video}' not found in the workspace.")
        sys.exit(1)
        
    print(f"[*] Source Video: {source_video} ({os.path.getsize(source_video) / (1024*1024):.2f} MB)")
    print("[*] Running inference and saving output video...")
    
    start_time = time.time()
    
    # Run prediction
    # stream=True processes the video as a generator to prevent memory issues for long videos
    results = model.predict(source=source_video, save=True, conf=0.25, device=device, stream=True)
    
    total_frames = 0
    fire_frames = 0
    smoke_frames = 0
    detections_summary = []
    
    print("\n[*] Processing frames:")
    for i, res in enumerate(results):
        total_frames += 1
        
        # Get count of detections for each class in the current frame
        classes_in_frame = res.boxes.cls.tolist()
        confidences = res.boxes.conf.tolist()
        
        frame_fire_count = classes_in_frame.count(0.0) # 0 corresponds to Fire based on data.yaml
        frame_smoke_count = classes_in_frame.count(2.0) # 2 corresponds to smoke
        
        if frame_fire_count > 0:
            fire_frames += 1
        if frame_smoke_count > 0:
            smoke_frames += 1
            
        if len(classes_in_frame) > 0:
            detections_summary.append((i, classes_in_frame, confidences))
            
        if total_frames % 50 == 0:
            print(f"    - Processed {total_frames} frames... (Current Detections in frame: Fire={frame_fire_count}, Smoke={frame_smoke_count})")
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("                 DETECTION & INFERENCE REPORT")
    print("=" * 60)
    print(f"[*] Total frames processed: {total_frames}")
    print(f"[*] Inference duration    : {elapsed_time:.2f} seconds")
    if total_frames > 0:
        print(f"[*] Average FPS           : {total_frames / elapsed_time:.2f}")
    print(f"[*] Frames containing FIRE: {fire_frames} ({fire_frames/total_frames*100:.1f}%)")
    print(f"[*] Frames containing SMOKE: {smoke_frames} ({smoke_frames/total_frames*100:.1f}%)")
    
    if len(detections_summary) > 0:
        all_confs = [c for s in detections_summary for c in s[2]]
        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
        max_conf = max(all_confs) if all_confs else 0.0
        print(f"[*] Average confidence    : {avg_conf * 100:.1f}%")
        print(f"[*] Maximum confidence    : {max_conf * 100:.1f}%")
    else:
        print("[*] No fire or smoke detected in the source video.")
        
    print("-" * 60)
    print("[*]Detections completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
