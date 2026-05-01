import cv2
import numpy as np
from ultralytics import YOLO
import pyvirtualcam
from pathlib import Path
import re

# ========== 1. 模型 ==========
model = YOLO("yolov8n-face.pt")

# ========== 2. 贴图 ==========
OVERLAY_PATH = "laughing_man_png_sequence_with_glitch/*.png"

def to_bgra(image):
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image

def natural_sort_key(path_obj):
    parts = re.split(r"(\d+)", path_obj.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def load_sequence(path_pattern):
    p = Path(path_pattern)
    files = sorted(p.parent.glob(p.name), key=natural_sort_key)
    frames = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        frames.append(to_bgra(img))
    return frames

overlay_frames = load_sequence(OVERLAY_PATH)

# ========== 3. 摄像头 ==========
cap = cv2.VideoCapture(0)

fps = 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ========== 4. 参数 ==========
OVERLAY_SIZE_SCALE = 3.0
OVERLAY_SEQUENCE_FPS = 30.0

frame_idx = 0

# ========== 5. 虚拟摄像头 ==========
with pyvirtualcam.Camera(width=w, height=h, fps=fps) as cam:
    print("Virtual cam:", cam.device)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== overlay frame =====
        overlay_i = int(frame_idx * OVERLAY_SEQUENCE_FPS / fps)
        overlay_frame = overlay_frames[overlay_i % len(overlay_frames)]
        frame_idx += 1

        # ===== YOLO tracking =====
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)

            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box.astype(int)

                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                gh, gw = overlay_frame.shape[:2]

                scale = min(bw/gw, bh/gh) * OVERLAY_SIZE_SCALE
                nw, nh = int(gw*scale), int(gh*scale)

                overlay = cv2.resize(overlay_frame, (nw, nh))

                cx = (x1 + x2)//2
                cy = (y1 + y2)//2

                ox1 = cx - nw//2
                oy1 = cy - nh//2
                ox2 = ox1 + nw
                oy2 = oy1 + nh

                fh, fw = frame.shape[:2]
                fx1, fy1 = max(0, ox1), max(0, oy1)
                fx2, fy2 = min(fw, ox2), min(fh, oy2)

                if fx1 >= fx2 or fy1 >= fy2:
                    continue

                sx1 = fx1 - ox1
                sy1 = fy1 - oy1
                sx2 = sx1 + (fx2 - fx1)
                sy2 = sy1 + (fy2 - fy1)

                overlay_crop = overlay[sy1:sy2, sx1:sx2]

                if overlay_crop.shape[2] == 4:
                    alpha = overlay_crop[:,:,3:4]/255.0
                    roi = frame[fy1:fy2, fx1:fx2]
                    frame[fy1:fy2, fx1:fx2] = (
                        alpha * overlay_crop[:,:,:3] +
                        (1-alpha) * roi
                    ).astype(np.uint8)

        # ===== 转 RGB 给虚拟摄像头 =====
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cam.send(frame_rgb)
        cam.sleep_until_next_frame()

cap.release()