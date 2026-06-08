import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
import os
import time
from PIL import Image
from pathlib import Path

# Import YOLOv5 model components
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.datasets import letterbox
from utils.plots import plot_one_box
from utils.torch_utils import select_device

# Streamlit Page Configuration
st.set_page_config(
    page_title="YOLO-CROWD Monitoring & Prevention System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Aesthetics
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0D1B2A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #1B7F8E;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .metric-container {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1B7F8E;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .alert-container-critical {
        background-color: #FDEDEC;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #E74C3C;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .alert-container-safe {
        background-color: #EAFAF1;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2ECC71;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .alert-text-crit {
        font-size: 1.2rem;
        font-weight: bold;
        color: #922B21;
    }
    .alert-text-safe {
        font-size: 1.2rem;
        font-weight: bold;
        color: #196F3D;
    }
    </style>
""", unsafe_allow_html=True)

# App Title & Layout
st.markdown("<div class='main-title'>🚨 YOLO-CROWD Monitoring System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-Time Crowd Counting & Stampede Prevention Dashboard</div>", unsafe_allow_html=True)

# ----------------- Load YOLO Model -----------------
@st.cache_resource
def load_model(weights_path="yolo-crowd.pt"):
    device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weights_path, map_location=device)
    model.eval()  # Eval mode
    return model, device

try:
    model, device = load_model("yolo-crowd.pt")
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# ----------------- Sidebar Controls -----------------
st.sidebar.header("⚙️ System Configuration")

if model_loaded:
    st.sidebar.success("✅ Model: YOLO-CROWD Loaded")
else:
    st.sidebar.error("❌ Model Loading Failed")

# People Limit Setting
st.sidebar.subheader("🚨 Threshold Settings")
count_limit = st.sidebar.slider("Maximum Allowed People Limit", min_value=1, max_value=100, value=10, step=1)

# Confidence and NMS Controls
st.sidebar.subheader("🔍 Detection Settings")
conf_thres = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
iou_thres = st.sidebar.slider("NMS IoU Threshold", min_value=0.05, max_value=1.0, value=0.45, step=0.05)
frame_skip = st.sidebar.slider("Frame Skip Rate", min_value=1, max_value=10, value=2, step=1, help="Process every N-th frame to skip redundant frames and speed up inference")

# Input Mode Selector
input_mode = st.sidebar.selectbox("Select Input Source", ("Demo Video", "Image Upload", "Video Upload"))

# Video Looping Option
loop_video = False
if input_mode in ("Demo Video", "Video Upload"):
    loop_video = st.sidebar.checkbox("Loop Video Stream", value=True, help="Continuously loop the video playback to simulate a live stream.")

# Model dimensions
stride = 32  # Enforce 32 to satisfy backbone downsampling requirements and avoid shape mismatches
img_size = 640  # Standard input resolution

# ----------------- Inference Helper -----------------
def run_inference(image_np, model, device, conf_thres, iou_thres, stride, img_size=640):
    """
    Runs YOLO-CROWD inference on a single numpy RGB image.
    Returns: count, annotated_image
    """
    h0, w0 = image_np.shape[:2]
    
    # 1. Image preprocessing (Resize and Pad)
    # letterbox takes a BGR image by default, but we can pass RGB since letterbox doesn't color swap
    img_padded, ratio, (dw, dh) = letterbox(image_np, new_shape=img_size, auto=True, stride=stride)
    
    # BGR to RGB (if input was BGR, but we use RGB for Streamlit)
    # Convert HWC to CHW, then to Tensor
    img_tensor = img_padded.transpose(2, 0, 1)
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor).to(device)
    
    # Normalization (0-255 to 0.0-1.0) and half precision if GPU
    half = device.type != 'cpu'
    if half:
        model.half()
        img_tensor = img_tensor.half()
    else:
        model.float()
        img_tensor = img_tensor.float()
        
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        
    # 2. Model Prediction
    with torch.no_grad():
        pred = model(img_tensor)[0]
        
    # 3. Non-Maximum Suppression (NMS)
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    # 4. Process results
    annotated_img = image_np.copy()
    count = 0
    
    det = pred[0]
    if len(det):
        # Rescale boxes from img_padded to original size
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], image_np.shape).round()
        count = len(det)
        
        # Draw bounding boxes (Draw green boxes)
        for *xyxy, conf, cls in det:
            # We draw on RGB image directly. Red/Coral color for detection box
            color = (255, 99, 71)  # Coral RGB
            label = f"Face {conf:.2f}"
            plot_one_box(xyxy, annotated_img, label=None, color=color, line_thickness=2)
            
    return count, annotated_img

# ----------------- App Main Logic -----------------
if not model_loaded:
    st.warning("Please ensure 'yolo-crowd.pt' is in the project directory.")
else:
    if input_mode == "Image Upload":
        st.subheader("🖼️ Static Image Crowd Analysis")
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            
            with st.spinner("Analyzing image..."):
                count, annotated_img = run_inference(image_np, model, device, conf_thres, iou_thres, stride, img_size)
            
            # Display Alerts/Aesthetics
            if count > count_limit:
                st.markdown(f"""
                    <div class="alert-container-critical">
                        <div class="alert-text-crit">🚨 CRITICAL ALERT: OCCUPANCY LIMIT EXCEEDED!</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="alert-container-safe">
                        <div class="alert-text-safe">✅ Safe Occupancy Zone</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Display Dashboard Metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(
                    label="Current Occupants",
                    value=count,
                    delta=f"{count - count_limit} over limit" if count > count_limit else f"{count_limit - count} remaining",
                    delta_color="inverse" if count > count_limit else "normal"
                )
            with col_m2:
                st.metric(label="Safety Limit", value=count_limit)
            
            # Display Images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Uploaded Image", use_container_width=True)
            with col2:
                st.image(annotated_img, caption="Detections (YOLO-CROWD)", use_container_width=True)

    elif input_mode == "Video Upload" or input_mode == "Demo Video":
        if input_mode == "Demo Video":
            st.subheader("📹 Demo Video Crowd Analysis")
            video_path = "test22.mp4"
            if not os.path.exists(video_path):
                st.error(f"Demo video file '{video_path}' not found in the workspace.")
                st.info("Please choose 'Video Upload' instead.")
                st.stop()
        else:
            st.subheader("📹 Video Upload Crowd Analysis")
            uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
            
            if uploaded_video is not None:
                # Save to a temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                video_path = tfile.name
            else:
                st.info("Please upload an MP4/AVI video file to start.")
                st.stop()
                
        # Load Video Capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video stream.")
            st.stop()
            
        # Video Stats
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Total Video Frames: {total_frames}")
        
        # Display Controls and Outputs
        start_btn = st.button("▶️ Run Video Analysis")
        stop_btn = st.sidebar.button("⏹️ Stop Stream")
        
        # Display Areas (Responsive Layout)
        col_video, col_stats = st.columns([3, 2])
        with col_video:
            video_placeholder = st.empty()
        with col_stats:
            alert_placeholder = st.empty()
            metric_placeholder = st.empty()
            chart_placeholder = st.empty()
            
        if start_btn:
            progress_bar = st.progress(0)
            frame_idx = 0
            counts_history = []
            
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    if loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                    
                frame_idx += 1
                
                # Skip redundant frames to speed up processing
                if frame_skip > 1 and frame_idx % frame_skip != 1:
                    if frame_idx % 5 == 0 or frame_idx == total_frames:
                        progress_val = min(float(frame_idx) / total_frames, 1.0)
                        progress_bar.progress(progress_val)
                    continue

                # Convert BGR frame from OpenCV to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run Model
                count, annotated_frame = run_inference(frame_rgb, model, device, conf_thres, iou_thres, stride, img_size)
                counts_history.append(count)
                
                # Update Video Display
                video_placeholder.image(annotated_frame, caption=f"Processing Frame {frame_idx}/{total_frames}", use_container_width=True)
                
                # Update Progress
                progress_val = min(float(frame_idx) / total_frames, 1.0)
                progress_bar.progress(progress_val)
                
                # Update Metrics Dashboard
                with metric_placeholder.container():
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric(
                            label="Current Occupants",
                            value=count,
                            delta=f"{count - count_limit} over limit" if count > count_limit else f"{count_limit - count} remaining",
                            delta_color="inverse" if count > count_limit else "normal"
                        )
                    with col_m2:
                        st.metric(label="Safety Limit", value=count_limit)
                
                # Live Warning Logic
                if count > count_limit:
                    alert_placeholder.markdown(f"""
                        <div class="alert-container-critical">
                            <div class="alert-text-crit">🚨 CRITICAL WARNING</div>
                            <div style="color: #922B21; font-size: 1.1rem; margin-top: 5px;">
                                Occupancy limit exceeded!
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    alert_placeholder.markdown(f"""
                        <div class="alert-container-safe">
                            <div class="alert-text-safe">✅ Zone Safe</div>
                            <div style="color: #196F3D; font-size: 1.1rem; margin-top: 5px;">
                                Occupancy is within limits.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Update Stats Graph
                if frame_idx % 5 == 0 or frame_idx == total_frames:
                    chart_placeholder.line_chart(counts_history)
                    
                # Small sleep to match frame rates
                time.sleep(0.01)
                
            cap.release()
            st.success("Video processing completed.")
            
            # Save stats if uploaded video temp file was created
            if input_mode == "Video Upload" and 'tfile' in locals():
                try:
                    os.unlink(tfile.name)
                except Exception:
                    pass
