import os
import sys
sys.path.append(os.path.abspath('.'))

import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox

# Load model
device = torch.device('cpu')
model = attempt_load("yolo-crowd.pt", map_location=device)
model.eval()

# Load video and read first frame
cap = cv2.VideoCapture('test22.mp4')
ret, frame = cap.read()
cap.release()

# Preprocess image
img_padded, ratio, (dw, dh) = letterbox(frame, new_shape=640, auto=True, stride=32)
img_tensor = img_padded.transpose(2, 0, 1)
img_tensor = np.ascontiguousarray(img_tensor) if 'np' in locals() else img_padded.transpose(2, 0, 1)
img_tensor = torch.from_numpy(img_tensor).to(device).float() / 255.0
img_tensor = img_tensor.unsqueeze(0)

print(f"Input tensor shape: {img_tensor.shape}")

# Custom forward pass with trace print
x = img_tensor
y = []
for i, m in enumerate(model.model):
    if m.f != -1:
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
    
    try:
        if isinstance(x, list):
            shapes_in = [t.shape for t in x]
            print(f"Layer {i} ({m.type}) inputs shapes: {shapes_in}")
        else:
            print(f"Layer {i} ({m.type}) input shape: {x.shape}")
            
        x = m(x)
        y.append(x if m.i in model.save else None)
        
        if isinstance(x, list):
            shapes_out = [t.shape for t in x]
            print(f"Layer {i} ({m.type}) output shapes: {shapes_out}")
        else:
            print(f"Layer {i} ({m.type}) output shape: {x.shape}")
            
    except Exception as e:
        print(f"Error at Layer {i} ({m.type}): {e}")
        # If input was a list, print sizes of tensors
        if isinstance(x, list):
            for idx, t in enumerate(x):
                print(f"  Tensor {idx} shape: {t.shape}")
        raise e
