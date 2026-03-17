import cv2
import time
import sys

print("Testing OpenCV webcam access on macOS...")

# Try AVFoundation first, then default
backends = [(cv2.CAP_AVFOUNDATION, "AVFoundation"), (cv2.CAP_ANY, "Default")]
cap = None

for backend, name in backends:
    print(f"Trying backend: {name}")
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        print(f"✅ Successfully opened camera with {name} backend")
        break
    cap.release()
    cap = None

if cap is None or not cap.isOpened():
    print("❌ Failed to open camera.")
    sys.exit(1)

# Do NOT set resolution to let it use default

print("Warming up camera...")
for i in range(10):
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"Frame {i}: Read successfully, shape = {frame.shape}, sum = {frame.sum()}")
    else:
        print(f"Frame {i}: Failed or None")
time.sleep(0.5)

print("Starting video feed. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to grab frame")
        break
        
    cv2.imshow('Test Webcam (Press Q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
