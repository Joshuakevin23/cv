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

# Initialize Session State for Persistent Analytics
if "counts_history" not in st.session_state:
    st.session_state.counts_history = []
if "peak_occupancy" not in st.session_state:
    st.session_state.peak_occupancy = 0
if "total_violations" not in st.session_state:
    st.session_state.total_violations = 0
if "violation_logs" not in st.session_state:
    st.session_state.violation_logs = []
if "current_count" not in st.session_state:
    st.session_state.current_count = 0

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
input_mode = st.sidebar.selectbox("Select Input Source", ("Demo Video", "Image Upload", "Video Upload", "IP Camera (RTSP)"))

# IP Camera RTSP Config
rtsp_url = ""
if input_mode == "IP Camera (RTSP)":
    st.sidebar.subheader("🔌 IP Camera Settings")
    rtsp_url = st.sidebar.text_input(
        "RTSP Stream URL",
        value="rtsp://admin:admin123@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1",
        help="Format: rtsp://username:password@ip_address:port/cam/realmonitor?channel=1&subtype=1"
    )
    st.sidebar.info(
        "💡 **CP Plus Format Guide**:\n"
        "`rtsp://<user>:<pass>@<ip>:554/cam/realmonitor?channel=1&subtype=1`\n\n"
        "*Note: subtype=1 targets the sub-stream for lower latency & faster inference.*"
    )

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
    img_padded, ratio, (dw, dh) = letterbox(image_np, new_shape=img_size, auto=True, stride=stride)
    
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
        
        # Draw bounding boxes (Draw Coral boxes)
        for *xyxy, conf, cls in det:
            color = (255, 99, 71)  # Coral RGB
            plot_one_box(xyxy, annotated_img, label=None, color=color, line_thickness=2)
            
    return count, annotated_img

# ----------------- App Main Logic -----------------
# Create Tabs for Separate Screens
tab_monitor, tab_stats = st.tabs(["🎥 Active Security Stream", "📊 Dashboard & Analytics"])

# 1. Dashboard Stats Screen (rendered first so placeholders exist in the context)
with tab_stats:
    st.subheader("📊 Session Analytics & Incident Logs")
    
    # Structural placeholders for live updates from the video loop
    stats_metrics_placeholder = st.empty()
    stats_chart_placeholder = st.empty()
    stats_log_placeholder = st.empty()
    
    # If the video is not actively running, render the static state from session state
    if st.session_state.counts_history:
        with stats_metrics_placeholder.container():
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            curr = st.session_state.current_count
            with col_m1:
                st.metric(
                    label="Last Count",
                    value=curr,
                    delta=f"{curr - count_limit} over limit" if curr > count_limit else f"{count_limit - curr} remaining",
                    delta_color="inverse" if curr > count_limit else "normal"
                )
            with col_m2:
                st.metric(label="Safety Limit", value=count_limit)
            with col_m3:
                st.metric(label="Peak Occupancy", value=st.session_state.peak_occupancy)
            with col_m4:
                st.metric(label="Total Violations (Frames)", value=st.session_state.total_violations)
                
        # Line chart of occupancy trend
        stats_chart_placeholder.markdown("### 📈 Occupancy Trend History")
        stats_chart_placeholder.line_chart(st.session_state.counts_history)
        
        # Incident log
        stats_log_placeholder.markdown("### 🚨 Safety Warning Incident Log")
        if st.session_state.violation_logs:
            stats_log_placeholder.dataframe(st.session_state.violation_logs)
        else:
            stats_log_placeholder.success("✅ No safety limit violations recorded in this session. All clear!")
    else:
        stats_metrics_placeholder.info("📊 No Active Monitoring Data. Please go to the 'Active Security Stream' tab and connect the camera feed or analyze an image/video to view analytics.")

# 2. Live Monitoring Screen
with tab_monitor:
    if not model_loaded:
        st.warning("Please ensure 'yolo-crowd.pt' is in the project directory.")
    else:
        if input_mode == "Image Upload":
            st.subheader("🖼️ Static Capture Analysis")
            uploaded_file = st.file_uploader("Upload static camera capture...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(image)
                
                with st.spinner("Running model inference..."):
                    count, annotated_img = run_inference(image_np, model, device, conf_thres, iou_thres, stride, img_size)
                
                # Update session state for stats tab
                st.session_state.counts_history = [count]
                st.session_state.peak_occupancy = count
                st.session_state.current_count = count
                if count > count_limit:
                    st.session_state.total_violations = 1
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    st.session_state.violation_logs = [{
                        "Time/Frame": f"Capture ({timestamp})",
                        "Occupant Count": count,
                        "Safety Limit": count_limit,
                        "Excess": count - count_limit,
                        "Status": "🚨 Alert"
                    }]
                else:
                    st.session_state.total_violations = 0
                    st.session_state.violation_logs = []
                
                # Display Alert status
                if count > count_limit:
                    st.markdown(f"""
                        <div class="alert-container-critical">
                            <div class="alert-text-crit">🚨 CRITICAL WARNING: OCCUPANCY LIMIT EXCEEDED!</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="alert-container-safe">
                            <div class="alert-text-safe">✅ ZONE STATUS: SAFE</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Quick stats on the monitoring screen
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric(
                        label="Detections",
                        value=count,
                        delta=f"{count - count_limit} over limit" if count > count_limit else f"{count_limit - count} remaining",
                        delta_color="inverse" if count > count_limit else "normal"
                    )
                with col_m2:
                    st.metric(label="Safety Limit", value=count_limit)
                
                # Show images
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Capture Feed", use_container_width=True)
                with col2:
                    st.image(annotated_img, caption="Processed Detections", use_container_width=True)

        elif input_mode == "Video Upload" or input_mode == "Demo Video":
            if input_mode == "Demo Video":
                st.subheader("📹 CCTV Feed Stream (Demo Video)")
                video_path = "test22.mp4"
                if not os.path.exists(video_path):
                    st.error(f"Demo video file '{video_path}' not found in the workspace.")
                    st.info("Please choose 'Video Upload' instead.")
                    st.stop()
            else:
                st.subheader("📹 Uploaded CCTV Feed Stream")
                uploaded_video = st.file_uploader("Upload surveillance footage...", type=["mp4", "avi", "mov"])
                
                if uploaded_video is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_video.read())
                    video_path = tfile.name
                else:
                    st.info("Please upload an MP4/AVI file to stream.")
                    st.stop()
                    
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Failed to connect to video feed.")
                st.stop()
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Streaming Control Buttons
            col_b1, col_b2 = st.columns([1, 4])
            with col_b1:
                start_btn = st.button("▶️ Connect Stream")
            with col_b2:
                stop_btn = st.button("⏹️ Disconnect Stream")
                
            # Placeholders for Stream Feed and Alert Status
            video_placeholder = st.empty()
            alert_placeholder = st.empty()
            progress_bar = st.empty()
            
            # Display status when stream is idle
            if not start_btn:
                video_placeholder.info("Click 'Connect Stream' above to start flowing in the camera stream.")
                
            if start_btn:
                # Reset metrics for the new active session
                st.session_state.counts_history = []
                st.session_state.peak_occupancy = 0
                st.session_state.total_violations = 0
                st.session_state.violation_logs = []
                st.session_state.current_count = 0
                
                frame_idx = 0
                progress_bar = st.progress(0)
                
                while cap.isOpened() and not stop_btn:
                    ret, frame = cap.read()
                    if not ret:
                        if loop_video:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            break
                            
                    frame_idx += 1
                    
                    # Skip frames to speed up CPU inference
                    if frame_skip > 1 and frame_idx % frame_skip != 1:
                        if frame_idx % 5 == 0 or frame_idx == total_frames:
                            progress_val = min(float(frame_idx) / total_frames, 1.0)
                            progress_bar.progress(progress_val)
                        continue
                        
                    # Convert BGR (OpenCV) to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run YOLO-CROWD Inference
                    count, annotated_frame = run_inference(frame_rgb, model, device, conf_thres, iou_thres, stride, img_size)
                    
                    # Update State
                    st.session_state.counts_history.append(count)
                    st.session_state.current_count = count
                    if count > st.session_state.peak_occupancy:
                        st.session_state.peak_occupancy = count
                        
                    # Log Safety Exceedance Event
                    if count > count_limit:
                        st.session_state.total_violations += 1
                        timestamp = time.strftime("%H:%M:%S", time.localtime())
                        st.session_state.violation_logs.append({
                            "Time/Frame": f"Frame {frame_idx} ({timestamp})",
                            "Occupant Count": count,
                            "Safety Limit": count_limit,
                            "Excess": count - count_limit,
                            "Status": "🚨 Critical"
                        })
                        
                    # Update Live View (in tab_monitor)
                    video_placeholder.image(
                        annotated_frame, 
                        caption=f"🎥 Live CCTV Stream (Frame {frame_idx}/{total_frames})", 
                        use_container_width=True
                    )
                    
                    # Update Alert Banner
                    if count > count_limit:
                        alert_placeholder.markdown(f"""
                            <div class="alert-container-critical">
                                <div class="alert-text-crit">🚨 SAFETY ALERT: OCCUPANCY LIMIT EXCEEDED</div>
                                <div style="color: #922B21; font-weight: 500; font-size: 1.1rem; margin-top: 5px;">
                                    Detected {count} people in zone (Limit: {count_limit}). Crowding warning active.
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        alert_placeholder.markdown(f"""
                            <div class="alert-container-safe">
                                <div class="alert-text-safe">🟢 SYSTEM ONLINE: SAFE OCCUPANCY</div>
                                <div style="color: #196F3D; font-weight: 500; font-size: 1.1rem; margin-top: 5px;">
                                    Detected {count} people in zone (Limit: {count_limit}). Safe status.
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    # Update Progress
                    progress_val = min(float(frame_idx) / total_frames, 1.0)
                    progress_bar.progress(progress_val)
                    
                    # Update Placeholders in tab_stats (Dashboard & Analytics)
                    with stats_metrics_placeholder.container():
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric(
                                label="Current Count",
                                value=count,
                                delta=f"{count - count_limit} over limit" if count > count_limit else f"{count_limit - count} remaining",
                                delta_color="inverse" if count > count_limit else "normal"
                            )
                        with col_m2:
                            st.metric(label="Safety Limit", value=count_limit)
                        with col_m3:
                            st.metric(label="Peak Occupancy", value=st.session_state.peak_occupancy)
                        with col_m4:
                            st.metric(label="Total Violations (Frames)", value=st.session_state.total_violations)
                            
                    stats_chart_placeholder.markdown("### 📈 Occupancy Trend History")
                    stats_chart_placeholder.line_chart(st.session_state.counts_history)
                    
                    stats_log_placeholder.markdown("### 🚨 Safety Warning Incident Log")
                    if st.session_state.violation_logs:
                        stats_log_placeholder.dataframe(st.session_state.violation_logs[-10:])
                    else:
                        stats_log_placeholder.info("No safety violations recorded in this session.")
                        
                    # Minor delay
                    time.sleep(0.01)
                    
                cap.release()
                st.success("Camera stream session closed.")
                
                # Cleanup
                if input_mode == "Video Upload" and 'tfile' in locals():
                    try:
                        os.unlink(tfile.name)
                    except Exception:
                        pass

        elif input_mode == "IP Camera (RTSP)":
            st.subheader("📹 Live IP Camera Stream (RTSP)")
            
            if not rtsp_url:
                st.info("Please configure a valid RTSP Stream URL in the sidebar settings.")
                st.stop()
                
            # Streaming Control Buttons
            col_b1, col_b2 = st.columns([1, 4])
            with col_b1:
                start_btn = st.button("▶️ Connect Camera Feed")
            with col_b2:
                stop_btn = st.button("⏹️ Disconnect Camera Feed")
                
            # Placeholders for Stream Feed and Alert Status
            video_placeholder = st.empty()
            alert_placeholder = st.empty()
            connection_status = st.empty()
            
            # Display status when stream is idle
            if not start_btn:
                video_placeholder.info("Click 'Connect Camera Feed' above to start flowing in the IP Camera stream.")
                
            if start_btn:
                # Reset metrics for the new active session
                st.session_state.counts_history = []
                st.session_state.peak_occupancy = 0
                st.session_state.total_violations = 0
                st.session_state.violation_logs = []
                st.session_state.current_count = 0
                
                connection_status.markdown("⚡ *Connecting to IP Camera stream...*")
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                
                if not cap.isOpened():
                    connection_status.error("❌ Failed to connect to RTSP stream. Please check camera connection, IP address, and credentials.")
                    st.stop()
                    
                connection_status.success("🟢 Camera Feed Connected successfully.")
                frame_idx = 0
                retry_count = 0
                max_retries = 10
                
                while cap.isOpened() and not stop_btn:
                    ret, frame = cap.read()
                    if not ret:
                        retry_count += 1
                        connection_status.warning(f"⚠️ Video stream frame dropped. Attempting to reconnect ({retry_count}/{max_retries})...")
                        time.sleep(1.0)
                        
                        cap.release()
                        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                        
                        if retry_count >= max_retries:
                            connection_status.error("❌ Live Stream Disconnected. Reconnection limit exceeded.")
                            break
                        continue
                        
                    retry_count = 0
                    connection_status.success("🟢 Live Stream Active")
                    frame_idx += 1
                    
                    # Skip frames to prevent inference lag on live stream
                    if frame_skip > 1 and frame_idx % frame_skip != 1:
                        continue
                        
                    # Convert BGR (OpenCV) to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run YOLO-CROWD Inference
                    count, annotated_frame = run_inference(frame_rgb, model, device, conf_thres, iou_thres, stride, img_size)
                    
                    # Update State
                    st.session_state.counts_history.append(count)
                    st.session_state.current_count = count
                    if count > st.session_state.peak_occupancy:
                        st.session_state.peak_occupancy = count
                        
                    # Log Safety Exceedance Event
                    if count > count_limit:
                        st.session_state.total_violations += 1
                        timestamp = time.strftime("%H:%M:%S", time.localtime())
                        st.session_state.violation_logs.append({
                            "Time/Frame": f"Stream Time: {timestamp} (Frame {frame_idx})",
                            "Occupant Count": count,
                            "Safety Limit": count_limit,
                            "Excess": count - count_limit,
                            "Status": "🚨 Critical Warning"
                        })
                        
                    # Update Live View (in tab_monitor)
                    video_placeholder.image(
                        annotated_frame, 
                        caption=f"🎥 Live IP Camera Feed (Frame {frame_idx})", 
                        use_container_width=True
                    )
                    
                    # Update Alert Banner
                    if count > count_limit:
                        alert_placeholder.markdown(f"""
                            <div class="alert-container-critical">
                                <div class="alert-text-crit">🚨 SAFETY ALERT: OCCUPANCY LIMIT EXCEEDED</div>
                                <div style="color: #922B21; font-weight: 500; font-size: 1.1rem; margin-top: 5px;">
                                    Detected {count} people in zone (Limit: {count_limit}). Crowding warning active!
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        alert_placeholder.markdown(f"""
                            <div class="alert-container-safe">
                                <div class="alert-text-safe">🟢 SYSTEM ONLINE: SAFE OCCUPANCY</div>
                                <div style="color: #196F3D; font-weight: 500; font-size: 1.1rem; margin-top: 5px;">
                                    Detected {count} people in zone (Limit: {count_limit}). Safe status.
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    # Update Placeholders in tab_stats (Dashboard & Analytics)
                    with stats_metrics_placeholder.container():
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric(
                                label="Current Count",
                                value=count,
                                delta=f"{count - count_limit} over limit" if count > count_limit else f"{count_limit - count} remaining",
                                delta_color="inverse" if count > count_limit else "normal"
                            )
                        with col_m2:
                            st.metric(label="Safety Limit", value=count_limit)
                        with col_m3:
                            st.metric(label="Peak Occupancy", value=st.session_state.peak_occupancy)
                        with col_m4:
                            st.metric(label="Total Violations (Frames)", value=st.session_state.total_violations)
                            
                    stats_chart_placeholder.markdown("### 📈 Occupancy Trend History")
                    stats_chart_placeholder.line_chart(st.session_state.counts_history)
                    
                    stats_log_placeholder.markdown("### 🚨 Safety Warning Incident Log")
                    if st.session_state.violation_logs:
                        stats_log_placeholder.dataframe(st.session_state.violation_logs[-10:])
                    else:
                        stats_log_placeholder.info("No safety violations recorded in this session.")
                        
                    # Tiny delay
                    time.sleep(0.005)
                    
                cap.release()
                connection_status.info("🔌 Live stream disconnected.")

