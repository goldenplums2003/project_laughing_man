import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import re

# ========== 1. 加载 YOLOv8-face ==========
model = YOLO("yolov8n-face.pt")


# ========== 2. 贴图素材（支持 PNG / PNG 序列） ==========
# 支持三种写法：
# 1) 单张 PNG: "laughing_man.png"
# 2) PNG 目录: "overlay_frames"
# 3) PNG 通配符: "overlay_frames/*.png"
OVERLAY_PATH = "laughing_man_png_sequence/*.png"


def to_bgra(image):
    if image is None:
        raise ValueError("overlay frame is empty")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    if image.shape[2] == 4:
        return image
    raise ValueError("Unsupported overlay channel count")


def natural_sort_key(path_obj):
    parts = re.split(r"(\d+)", path_obj.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def load_png_sequence(image_paths):
    frames = []
    for image_path in image_paths:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot load overlay image: {image_path}")
        frames.append(to_bgra(img))
    if not frames:
        raise FileNotFoundError("PNG sequence is empty")
    return frames


overlay_mode = None
overlay_path = Path(OVERLAY_PATH)
overlay_path_str = str(overlay_path)

if any(c in overlay_path_str for c in "*?[]"):
    overlay_mode = "sequence"
    sequence_parent = overlay_path.parent if str(overlay_path.parent) not in {"", "."} else Path(".")
    sequence_pattern = overlay_path.name
    sequence_files = sorted(
        [
            p for p in sequence_parent.glob(sequence_pattern)
            if p.is_file() and p.suffix.lower() == ".png"
        ],
        key=natural_sort_key
    )
    if not sequence_files:
        raise FileNotFoundError(f"No PNG files matched: {OVERLAY_PATH}")
    overlay_frames = load_png_sequence(sequence_files)
elif overlay_path.is_dir():
    overlay_mode = "sequence"
    sequence_files = sorted(
        [
            p for p in overlay_path.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        ],
        key=natural_sort_key
    )
    if not sequence_files:
        raise FileNotFoundError(f"No PNG files found in directory: {OVERLAY_PATH}")
    overlay_frames = load_png_sequence(sequence_files)
elif overlay_path.suffix.lower() == ".png":
    overlay_mode = "image"
    static_overlay = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
    if static_overlay is None:
        raise FileNotFoundError(f"Cannot load overlay image: {overlay_path}")
    static_overlay = to_bgra(static_overlay)
else:
    raise ValueError("OVERLAY_PATH only supports PNG file or PNG sequence")

# ========== 3. 输入/输出 ==========
INPUT_DIR = Path("tests")
OUTPUT_DIR = Path("outputs")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
    raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

input_files = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in VIDEO_EXTS
)

if not input_files:
    raise FileNotFoundError(f"No video files found in: {INPUT_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 可选：只贴某些 ID
TARGET_IDS = None

# 1) 覆盖物大小：
# keep_source 模式下：1.0 表示等比例“正好放进框内”；
# follow_box 模式下：1.0 表示“与框同宽同高”。
OVERLAY_SIZE_SCALE = 3.5

# 2) 长宽比模式：
# - "keep_source": 保持覆盖物原始长宽比，不拉伸
# - "follow_box": 跟随检测框长宽比，允许拉伸
OVERLAY_ASPECT_MODE = "keep_source"

# 3) PNG 序列播放速度（固定每秒多少张，不跟输入视频 FPS 走）
OVERLAY_SEQUENCE_FPS = 30.0

# 4) 调试显示：输出视频里标注每张脸的跟踪 ID
SHOW_TRACK_ID = False
SHOW_TRACK_BOX = False

if OVERLAY_SIZE_SCALE <= 0:
    raise ValueError("OVERLAY_SIZE_SCALE must be > 0")

if OVERLAY_ASPECT_MODE not in {"keep_source", "follow_box"}:
    raise ValueError("OVERLAY_ASPECT_MODE must be 'keep_source' or 'follow_box'")

if OVERLAY_SEQUENCE_FPS <= 0:
    raise ValueError("OVERLAY_SEQUENCE_FPS must be > 0")

# ========== 4. 主循环 ==========
for input_path in input_files:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"skip: cannot open input video: {input_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w <= 0 or h <= 0:
        print(f"skip: invalid video size: {input_path}")
        cap.release()
        continue

    output_path = OUTPUT_DIR / f"{input_path.stem}_output.mp4"
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    if not out.isOpened():
        print(f"skip: cannot open output video: {output_path}")
        cap.release()
        continue

    input_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if overlay_mode == "sequence":
            # 用固定序列 FPS + 输入时间戳决定当前序列帧
            overlay_frame_idx = int((input_frame_idx * OVERLAY_SEQUENCE_FPS) / fps)
            overlay_frame = overlay_frames[overlay_frame_idx % len(overlay_frames)]
        else:
            overlay_frame = static_overlay

        input_frame_idx += 1

        # ========= YOLO + ByteTrack =========
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        if results.boxes is not None and results.boxes.id is not None:

            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):

                x1, y1, x2, y2 = box.astype(int)

                # 在过滤前先显示 ID，方便确认该填哪些 TARGET_IDS
                if SHOW_TRACK_ID:
                    if SHOW_TRACK_BOX:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    text_y = y1 - 10 if y1 - 10 > 20 else y1 + 24
                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if TARGET_IDS and track_id not in TARGET_IDS:
                    continue

                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                gh, gw = overlay_frame.shape[:2]
                if gw <= 0 or gh <= 0:
                    continue

                if OVERLAY_ASPECT_MODE == "keep_source":
                    # 保持原图长宽比：先框内等比例适配，再乘大小系数
                    fit_scale = min(bw / gw, bh / gh)
                    draw_scale = fit_scale * OVERLAY_SIZE_SCALE
                    new_w = max(1, int(gw * draw_scale))
                    new_h = max(1, int(gh * draw_scale))
                else:
                    # 跟随框比例：直接按框宽高乘大小系数
                    new_w = max(1, int(bw * OVERLAY_SIZE_SCALE))
                    new_h = max(1, int(bh * OVERLAY_SIZE_SCALE))

                overlay = cv2.resize(overlay_frame, (new_w, new_h))

                # 以检测框中心点居中放置
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                ox1 = cx - new_w // 2
                oy1 = cy - new_h // 2
                ox2 = ox1 + new_w
                oy2 = oy1 + new_h

                # 只与画面边界求交，不限制在检测框内
                fh, fw = frame.shape[:2]
                fx1 = max(0, ox1)
                fy1 = max(0, oy1)
                fx2 = min(fw, ox2)
                fy2 = min(fh, oy2)

                if fx1 >= fx2 or fy1 >= fy2:
                    continue

                sx1 = fx1 - ox1
                sy1 = fy1 - oy1
                sx2 = sx1 + (fx2 - fx1)
                sy2 = sy1 + (fy2 - fy1)

                overlay_crop = overlay[sy1:sy2, sx1:sx2]

                if overlay_crop.shape[2] == 4:
                    alpha = (overlay_crop[:, :, 3:4] / 255.0).astype(np.float32)
                    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
                    blended = alpha * overlay_crop[:, :, :3].astype(np.float32) + (1 - alpha) * roi
                    frame[fy1:fy2, fx1:fx2] = blended.astype(np.uint8)

        out.write(frame)

    cap.release()
    out.release()
    print(f"done: {input_path} -> {output_path}")

print("all done")