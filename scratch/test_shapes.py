import os
import sys
sys.path.append(os.path.abspath('.'))

import torch
from models.experimental import attempt_load

# Load model
device = torch.device('cpu')
model = attempt_load("yolo-crowd.pt", map_location=device)
model.eval()

# Let's test a range of shapes that are multiples of 32
print("Testing multiples of 32 starting from 256:")
for h in range(256, 700, 32):
    for w in range(256, 700, 32):
        try:
            x = torch.zeros(1, 3, h, w)
            y = model(x)
            # print(f"Passed: {h}x{w}")
        except RuntimeError as e:
            print(f"Failed at input shape {h}x{w}: {e}")
            sys.exit(1)
print("All multiples of 32 from 256 to 700 passed!")
