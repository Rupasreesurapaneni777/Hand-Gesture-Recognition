from flask import Flask, Response, jsonify, render_template_string
import cv2
import numpy as np
import threading
import time
from GestureAPI import Gesture, DefineGestures, DecideGesture

app = Flask(__name__)

# ---------------- Configuration ----------------
HSV_THRESH_LOWER = 150
GAUSSIAN_KSIZE = 11
MORPH_ELEM_SIZE = 13
MEDIAN_KSIZE = 3
CAPTURE_BOX_COUNT = 9
CAPTURE_BOX_DIM = 20
CAPTURE_BOX_SEP_X = 8
CAPTURE_BOX_SEP_Y = 18
CAP_REGION_X_BEGIN = 0.5
CAP_REGION_Y_END = 0.8
FINGER_THRESH_L = 2.0
FINGER_THRESH_U = 3.8
RADIUS_THRESH = 0.04

# ---------------- Shared state ----------------
state_lock = threading.Lock()
camera = None
latest_frame = None
bg_model = None
hand_histogram = None
bg_captured = False
capture_done = False
last_gesture = "NONE"
last_finger_count = 0
first_iteration = True
finger_ct_history = [0, 0]

GestureDictionary = DefineGestures()
frame_gesture = Gesture("frame_gesture")


def get_camera():
    """Open the webcam lazily so importing app.py does not immediately use the camera."""
    global camera
    with state_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def capture_box_positions(frame):
    """Return 3x3 sample boxes positioned in the right-side detection region."""
    h, w = frame.shape[:2]
    start_x = int(w * 0.72)
    start_y = int(h * 0.28)

    xs, ys = [], []
    for row in range(3):
        for col in range(3):
            xs.append(start_x + col * (CAPTURE_BOX_DIM + CAPTURE_BOX_SEP_X))
            ys.append(start_y + row * (CAPTURE_BOX_DIM + CAPTURE_BOX_SEP_Y))
    return np.array(xs, dtype=int), np.array(ys, dtype=int)


def hand_capture(frame_in, box_x, box_y):
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    roi = np.zeros(
        [CAPTURE_BOX_DIM * CAPTURE_BOX_COUNT, CAPTURE_BOX_DIM, 3],
        dtype=hsv.dtype,
    )

    for i in range(CAPTURE_BOX_COUNT):
        roi[i * CAPTURE_BOX_DIM:(i + 1) * CAPTURE_BOX_DIM, 0:CAPTURE_BOX_DIM] = \
            hsv[box_y[i]:box_y[i] + CAPTURE_BOX_DIM,
                box_x[i]:box_x[i] + CAPTURE_BOX_DIM]

    hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def remove_bg(frame):
    global bg_model
    if bg_model is None:
        return frame
    fg_mask = bg_model.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    return cv2.bitwise_and(frame, frame, mask=fg_mask)


def hand_threshold(frame_in, hand_hist):
    frame_in = cv2.medianBlur(frame_in, 3)
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    # Only analyze the right-side detection region.
    hsv[0:int(CAP_REGION_Y_END * h), 0:int(CAP_REGION_X_BEGIN * w)] = 0
    hsv[int(CAP_REGION_Y_END * h):h, 0:w] = 0

    back_projection = cv2.calcBackProject(
        [hsv], [0, 1], hand_hist, [0, 180, 0, 256], 1
    )
    disc = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_ELEM_SIZE, MORPH_ELEM_SIZE)
    )
    cv2.filter2D(back_projection, -1, disc, back_projection)
    back_projection = cv2.GaussianBlur(
        back_projection, (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), 0
    )
    back_projection = cv2.medianBlur(back_projection, MEDIAN_KSIZE)
    _, thresh = cv2.threshold(back_projection, HSV_THRESH_LOWER, 255, 0)
    return thresh


def hand_contour_find(contours):
    if not contours:
        return False, None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return False, None
    return True, contour


def mark_hand_center(frame_in, contour):
    """Approximate the palm center using the strongest point inside the contour."""
    x, y, w, h = cv2.boundingRect(contour)
    max_d = 0.0
    pt = (0, 0)

    y1, y2 = int(y + 0.30 * h), int(y + 0.80 * h)
    x1, x2 = int(x + 0.30 * w), int(x + 0.60 * w)

    # Step by 2 pixels for smoother browser performance.
    for ind_y in range(y1, max(y1 + 1, y2), 2):
        for ind_x in range(x1, max(x1 + 1, x2), 2):
            dist = cv2.pointPolygonTest(contour, (ind_x, ind_y), True)
            if dist > max_d:
                max_d = dist
                pt = (ind_x, ind_y)

    score = max_d > RADIUS_THRESH * frame_in.shape[1]
    if score:
        cv2.circle(frame_in, pt, int(max_d), (255, 0, 0), 2)
    return frame_in, pt, max_d, score


def mark_fingers(frame_in, hull, pt, radius):
    global first_iteration, finger_ct_history, last_finger_count

    if hull is None or len(hull) < 2:
        return frame_in, [], [pt, radius]

    cx, cy = pt
    candidates = []

    # Keep sufficiently separated convex-hull points.
    for i in range(len(hull)):
        p1 = hull[i][0]
        p2 = hull[(i + 1) % len(hull)][0]
        dist = np.linalg.norm(p1 - p2)
        if dist > 18:
            candidates.append((int(p1[0]), int(p1[1])))

    # Remove duplicate / very close points.
    fingers = []
    for p in candidates:
        if not fingers or all(np.hypot(p[0] - q[0], p[1] - q[1]) > 20 for q in fingers):
            fingers.append(p)

    filtered = []
    for x, y in fingers:
        dist = np.hypot(x - cx, y - cy)
        if FINGER_THRESH_L * radius <= dist <= FINGER_THRESH_U * radius and y <= cy + radius:
            filtered.append((x, y))

    # Prefer the five highest points if noise creates more than five candidates.
    filtered = sorted(filtered, key=lambda p: p[1])[:5]

    if first_iteration:
        finger_ct_history[0] = finger_ct_history[1] = len(filtered)
        first_iteration = False
    else:
        finger_ct_history[0] = 0.34 * (
            finger_ct_history[0] + finger_ct_history[1] + len(filtered)
        )

    smoothed = int(round(finger_ct_history[0]))
    smoothed = max(0, min(5, smoothed))
    finger_ct_history[1] = len(filtered)
    last_finger_count = smoothed

    cv2.putText(
        frame_in,
        f"FINGERS: {smoothed}",
        (int(0.62 * frame_in.shape[1]), int(0.88 * frame_in.shape[0])),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for finger in filtered:
        cv2.circle(frame_in, finger, 8, (0, 255, 255), 2)
        cv2.line(frame_in, finger, (cx, cy), (255, 255, 255), 2)

    return frame_in, filtered, [(cx, cy), radius]


def find_gesture(frame_in, finger, palm):
    global last_gesture

    if not finger or palm[1] <= 0:
        last_gesture = "NONE"
        return frame_in, last_gesture

    frame_gesture.set_palm(palm[0], palm[1])
    frame_gesture.set_finger_pos(finger)
    frame_gesture.calc_angles()
    last_gesture = DecideGesture(frame_gesture, GestureDictionary)

    cv2.putText(
        frame_in,
        f"GESTURE: {last_gesture}",
        (int(0.53 * frame_in.shape[1]), int(0.96 * frame_in.shape[0])),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame_in, last_gesture


def draw_setup(frame):
    box_x, box_y = capture_box_positions(frame)
    for i in range(CAPTURE_BOX_COUNT):
        cv2.rectangle(
            frame,
            (int(box_x[i]), int(box_y[i])),
            (int(box_x[i] + CAPTURE_BOX_DIM), int(box_y[i] + CAPTURE_BOX_DIM)),
            (255, 0, 0),
            1,
        )
    return frame


def process_frame(frame):
    global latest_frame, last_gesture

    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    frame_original = frame.copy()

    h, w = frame.shape[:2]
    cv2.rectangle(
        frame,
        (int(CAP_REGION_X_BEGIN * w), 0),
        (w - 1, int(CAP_REGION_Y_END * h)),
        (255, 0, 0),
        2,
    )

    with state_lock:
        latest_frame = frame_original.copy()
        local_bg_captured = bg_captured
        local_capture_done = capture_done
        local_hist = hand_histogram

    if not local_bg_captured:
        frame = draw_setup(frame)
        cv2.putText(
            frame,
            "Step 1: Remove hand, then click Capture Background",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        last_gesture = "NONE"
        return frame

    fg_frame = remove_bg(frame_original)

    if not local_capture_done or local_hist is None:
        frame = draw_setup(frame)
        cv2.putText(
            frame,
            "Step 2: Cover all 9 boxes with your hand, then click Capture Hand",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        last_gesture = "NONE"
        return frame

    threshold = hand_threshold(fg_frame, local_hist)
    contours_info = cv2.findContours(
        threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    found, hand_contour = hand_contour_find(contours)
    if found:
        hand_hull = cv2.convexHull(hand_contour)
        frame, hand_center, hand_radius, hand_size_score = mark_hand_center(
            frame, hand_contour
        )
        if hand_size_score:
            frame, finger, palm = mark_fingers(
                frame, hand_hull, hand_center, hand_radius
            )
            frame, _ = find_gesture(frame, finger, palm)
        else:
            last_gesture = "NONE"
    else:
        last_gesture = "NONE"

    cv2.putText(
        frame,
        f"GESTURE: {last_gesture}",
        (int(0.53 * w), int(0.96 * h)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def generate_frames():
    cam = get_camera()
    if not cam.isOpened():
        return

    while True:
        success, frame = cam.read()
        if not success:
            time.sleep(0.1)
            continue

        frame = process_frame(frame)
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hand Gesture Recognition</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f4f7fb; color: #152238; }
    .wrap { max-width: 980px; margin: 30px auto; padding: 0 18px; }
    .card { background: white; border-radius: 16px; padding: 24px; box-shadow: 0 8px 28px rgba(0,0,0,.08); }
    h1 { margin-top: 0; }
    .video { width: 100%; max-width: 760px; border-radius: 12px; border: 2px solid #d8e0ea; display: block; margin: 18px auto; background: #111; }
    .buttons { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
    button { border: 0; padding: 11px 16px; border-radius: 9px; cursor: pointer; font-weight: 700; }
    .primary { background: #1f6feb; color: white; }
    .success { background: #238636; color: white; }
    .danger { background: #d1242f; color: white; }
    .status { margin: 18px auto 0; max-width: 760px; padding: 12px 14px; border-radius: 10px; background: #eef4ff; }
    .steps { line-height: 1.7; }
    code { background: #eef1f5; padding: 2px 6px; border-radius: 5px; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>🤚 Hand Gesture Recognition</h1>
    <p>OpenCV + Flask browser demo. Supported gestures: <b>V</b>, <b>L_right</b>, and <b>Index_Pointing</b>.</p>

    <div class="steps">
      <b>1.</b> Keep your hand outside the blue region and click <b>Capture Background</b>.<br>
      <b>2.</b> Place your hand over all 9 small boxes and click <b>Capture Hand</b>.<br>
      <b>3.</b> Show a supported gesture inside the blue detection region.
    </div>

    <img class="video" src="{{ url_for('video_feed') }}" alt="Webcam stream">

    <div class="buttons">
      <button class="primary" onclick="doAction('background')">Capture Background</button>
      <button class="success" onclick="doAction('hand')">Capture Hand</button>
      <button class="danger" onclick="doAction('reset')">Reset</button>
    </div>

    <div class="status" id="status">Status: waiting for camera...</div>
  </div>
</div>
<script>
async function doAction(name) {
  const r = await fetch('/action/' + name, {method: 'POST'});
  const data = await r.json();
  document.getElementById('status').textContent = data.message;
}
async function refreshStatus() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    document.getElementById('status').textContent =
      `Background: ${d.background ? 'captured' : 'not captured'} | ` +
      `Hand histogram: ${d.hand ? 'captured' : 'not captured'} | ` +
      `Fingers: ${d.fingers} | Gesture: ${d.gesture}`;
  } catch (_) {}
}
setInterval(refreshStatus, 1000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/action/background", methods=["POST"])
def capture_background():
    global bg_model, bg_captured, capture_done, hand_histogram
    global first_iteration, finger_ct_history, last_gesture, last_finger_count

    with state_lock:
        bg_model = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=25, detectShadows=False
        )
        bg_captured = True
        capture_done = False
        hand_histogram = None
        first_iteration = True
        finger_ct_history = [0, 0]
        last_gesture = "NONE"
        last_finger_count = 0

    return jsonify({
        "ok": True,
        "message": "Background capture started. Keep your hand outside the frame for a moment, then capture the hand histogram."
    })


@app.route("/action/hand", methods=["POST"])
def capture_hand():
    global hand_histogram, capture_done, first_iteration, finger_ct_history

    with state_lock:
        if not bg_captured:
            return jsonify({"ok": False, "message": "Capture the background first."}), 400
        if latest_frame is None:
            return jsonify({"ok": False, "message": "Camera frame is not ready yet."}), 400
        frame = latest_frame.copy()

    box_x, box_y = capture_box_positions(frame)
    hist = hand_capture(frame, box_x, box_y)

    with state_lock:
        hand_histogram = hist
        capture_done = True
        first_iteration = True
        finger_ct_history = [0, 0]

    return jsonify({
        "ok": True,
        "message": "Hand histogram captured. Now show a gesture inside the blue region."
    })


@app.route("/action/reset", methods=["POST"])
def reset():
    global bg_model, hand_histogram, bg_captured, capture_done
    global first_iteration, finger_ct_history, last_gesture, last_finger_count

    with state_lock:
        bg_model = None
        hand_histogram = None
        bg_captured = False
        capture_done = False
        first_iteration = True
        finger_ct_history = [0, 0]
        last_gesture = "NONE"
        last_finger_count = 0

    return jsonify({"ok": True, "message": "Reset complete. Start with background capture."})


@app.route("/status")
def status():
    with state_lock:
        return jsonify({
            "background": bg_captured,
            "hand": capture_done,
            "gesture": last_gesture,
            "fingers": last_finger_count,
        })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)