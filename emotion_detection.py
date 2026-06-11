import numpy as np
import cv2


EMOTION_INFO = {
    "happy":    {"symbol": ":)",  "mood": "Positive"},
    "surprise": {"symbol": ":O",  "mood": "Positive"},
    "neutral":  {"symbol": ":|",  "mood": "Normal"},
    "sad":      {"symbol": ":(",  "mood": "Stressed"},
    "angry":    {"symbol": ">:(", "mood": "Stressed"},
    "fear":     {"symbol": "D:",  "mood": "Stressed"},
    "disgust":  {"symbol": ":S",  "mood": "Stressed"},
}

MOOD_BG_RECIPE = {
    "Positive": "cheerful",
    "Normal":   "clean",
    "Stressed": "calm",
}


def build_mood_bg(height: int, width: int, style: str) -> np.ndarray:
    
    if style == "cheerful":
        top    = np.array([255, 140,  60], dtype=np.float32)
        bottom = np.array([255, 230,  80], dtype=np.float32)
    elif style == "clean":
        top    = np.array([180, 210, 240], dtype=np.float32)
        bottom = np.array([220, 225, 230], dtype=np.float32)
    else:
        top    = np.array([ 30, 140, 130], dtype=np.float32)
        bottom = np.array([100, 200, 170], dtype=np.float32)

    t    = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, np.newaxis]
    rows = (top * (1.0 - t) + bottom * t).astype(np.uint8)

    return np.tile(rows[:, np.newaxis, :], (1, width, 1))


class EmotionDetector:

    def __init__(self, run_every_n_frames: int = 5):
        self._run_every     = run_every_n_frames
        self._frame_counter = 0
        self._last_result   = None
        self._detector      = None
        self._available     = None

    def _try_load(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from fer.fer import FER
            self._detector  = FER(mtcnn=False)
            self._available = True
            print("[EmotionDetector] FER model loaded successfully.")
        except Exception as exc:
            self._available = False
            print(f"[EmotionDetector] FER not available — emotion overlay disabled. ({exc})")
        return self._available

    def _run_fer(self, rgb_frame: np.ndarray):
        try:
            bgr = rgb_frame[..., ::-1]
            results = self._detector.detect_emotions(bgr)

            if not results:
                return None

            face_data = max(results, key=lambda f: f["box"][2] * f["box"][3])
            emotions  = face_data["emotions"]

            dominant   = max(emotions, key=emotions.get)
            confidence = emotions[dominant] * 100.0

            info = EMOTION_INFO.get(dominant, {"symbol": "?", "mood": "Normal"})

            return {
                "emotion":    dominant,
                "confidence": confidence,
                "symbol":     info["symbol"],
                "mood":       info["mood"],
            }
        except Exception:
            return None

    def analyse(self, rgb_frame: np.ndarray):

        if not self._try_load():
            return None

        self._frame_counter += 1
        if self._frame_counter % self._run_every == 0:
            self._last_result = self._run_fer(rgb_frame)

        return self._last_result


def _draw_smiley(frame: np.ndarray, cx: int, cy: int, radius: int) -> None:
    yellow = (255, 220, 40)
    dark   = (40,  40,  40)

    cv2.circle(frame, (cx, cy), radius,     yellow, thickness=-1,         lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius,     dark,   thickness=max(2, radius//20), lineType=cv2.LINE_AA)

    eye_r  = max(3, radius // 7)
    eye_dy = radius // 3
    eye_dx = radius // 3
    cv2.circle(frame, (cx - eye_dx, cy - eye_dy), eye_r, dark, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx + eye_dx, cy - eye_dy), eye_r, dark, thickness=-1, lineType=cv2.LINE_AA)

    smile_r  = int(radius * 0.55)
    smile_cy = cy + radius // 8
    n_pts    = 20
    angles   = np.linspace(np.pi * 0.15, np.pi * 0.85, n_pts)
    pts = np.array(
        [[int(cx + smile_r * np.cos(a)),
          int(smile_cy + smile_r * np.sin(a))] for a in angles],
        dtype=np.int32
    )
    cv2.polylines(frame, [pts], isClosed=False, color=dark,
                  thickness=max(2, radius // 12), lineType=cv2.LINE_AA)


def apply_emotion_overlay(
    frame:    np.ndarray,
    seg_mask: np.ndarray | None,
    result:   dict | None,
    mood_bg:  np.ndarray | None,
) -> np.ndarray:
    output = frame.copy()
    h, w   = output.shape[:2]

    if seg_mask is not None and mood_bg is not None:
        person_mask = (seg_mask > 0.5)
        person_3ch  = np.stack([person_mask] * 3, axis=-1)
        output = np.where(person_3ch, frame, mood_bg).astype(np.uint8)

    if result is None:
        return output

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    panel_w  = 230
    panel_h  = 115
    margin   = 10

    smiley_clearance = 0
    if result.get("mood") == "Positive":
        smiley_r         = max(30, min(60, h // 9))
        smiley_cx        = w - smiley_r - 10
        smiley_cy        = smiley_r + 10
        _draw_smiley(canvas, smiley_cx, smiley_cy, smiley_r)
        smiley_clearance = smiley_r * 2 + margin * 2

    x0 = w - panel_w - margin
    y0 = margin + smiley_clearance
    x1 = w - margin
    y1 = y0 + panel_h

    cv2.rectangle(canvas, (x0, y0), (x1, y1), (30, 30, 30), thickness=-1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 80, 80), thickness=1)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    lh    = 25
    tx    = x0 + 8
    ty    = y0 + 22

    emotion_label = result["emotion"].capitalize()
    symbol        = result["symbol"]
    confidence    = result["confidence"]
    mood          = result["mood"]

    cv2.putText(canvas, f"Emotion: {emotion_label}  {symbol}",
                (tx, ty), font, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Confidence: {confidence:.0f}%",
                (tx, ty + lh), font, 0.52, white, 1, cv2.LINE_AA)

    mood_color = {
        "Positive": ( 80, 220,  80),
        "Normal":   ( 80, 160, 255),
        "Stressed": ( 80,  80, 220),
    }.get(mood, white)
    cv2.putText(canvas, f"Mood: {mood}",
                (tx, ty + 2 * lh), font, 0.52, mood_color, 1, cv2.LINE_AA)

    style = MOOD_BG_RECIPE.get(mood, "default")
    cv2.putText(canvas, f"BG: {style}",
                (tx, ty + 3 * lh), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    canvas = cv2.flip(canvas, 1)

    mask = canvas.any(axis=2)
    alpha = 0.85
    output[mask] = np.clip(
        canvas[mask].astype(np.float32) * alpha +
        output[mask].astype(np.float32) * (1 - alpha),
        0, 255
    ).astype(np.uint8)

    return output
