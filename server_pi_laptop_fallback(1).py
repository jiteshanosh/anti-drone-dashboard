import asyncio
import base64
import json
import os
import time
import warnings
from types import SimpleNamespace

# Silence optional inference models we don't use (must be set BEFORE importing inference).
for _flag in (
    "CORE_MODEL_GAZE_ENABLED", "PALIGEMMA_ENABLED", "FLORENCE2_ENABLED",
    "QWEN_2_5_ENABLED", "QWEN_3_ENABLED", "CORE_MODEL_SAM_ENABLED",
    "CORE_MODEL_SAM3_ENABLED", "SMOLVLM2_ENABLED", "DEPTH_ESTIMATION_ENABLED",
    "MOONDREAM2_ENABLED", "CORE_MODEL_TROCR_ENABLED", "CORE_MODEL_GROUNDINGDINO_ENABLED",
):
    os.environ.setdefault(_flag, "False")
warnings.filterwarnings("ignore", message=".*Specified provider.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
from inference import get_model
from ultralytics.trackers.byte_tracker import BYTETracker
import websockets


# Dashboard (browser on the same laptop) connects here.
HOST = "0.0.0.0"
PORT = 8765
STATUS_PORT = 8766   # dashboard Step 10 "hardware bridge" status feed

# Raspberry Pi connects here over the laptop's hotspot.
# 0.0.0.0 = listen on all adapters (including the hotspot adapter).
PI_HOST = "0.0.0.0"
PI_PORT = 8770
PI_STALE_SECONDS = 2.0

# ---- Gimbal control (commands sent back to the Pi) ----
# The gimbal is corrected in small, limited steps once per fresh detection
# result. Tune these after checking the model's measured FPS at startup.
GIMBAL_DEADZONE = 0.05        # ignore errors below this fraction of half-frame
PAN_GAIN_DEG = 5.0            # degrees of correction at full-frame error
TILT_GAIN_DEG = 5.0
MAX_STEP_DEG = 5.0            # cap per update to prevent overshoot
GIMBAL_SETTLE_SECONDS = 0.25  # ignore results from frames captured before the servos settled
# Direction. The Pi image is NOT mirrored (test.py mirrored it and used
# pan += turn_x), so pan is inverted here to keep the same physical behavior.
# If the gimbal moves away from the target, flip the sign of that axis.
PAN_SIGN = -1
TILT_SIGN = 1
DRONE_CLASSES = {"drone", "uav", "uas", "quadcopter", "multirotor", "hexacopter", "octocopter"}

CONFIDENCE = 0.45
JPEG_QUALITY = 55

LOCAL_MODEL_ID = "jitesh-a/aerial-object-detection-gbv9j-1-yolo26n-t1"

TRACK_HIGH_THRESH = 0.25
TRACK_LOW_THRESH = 0.10
NEW_TRACK_THRESH = 0.25
TRACK_BUFFER = 30
MATCH_THRESH = 0.80
FUSE_SCORE = True

clients = set()
status_clients = set()
TRACK_COLORS = {}

# Latest frame received from the Pi.
pi_state = {"frame": None, "id": 0, "t": 0.0, "connected": False, "ws": None}

# Fallback laptop webcam state. It is used only while the Raspberry Pi camera
# stream is not live. The Pi always has priority when its stream is fresh.
local_camera_state = {"connected": False, "t": 0.0, "source": "LAPTOP"}

# Judge/browser webcam state. The dashboard sends JPEG frames from the
# judge's browser over the same WebSocket. This lets remote judges use
# their own webcam instead of requiring the Raspberry Pi or the laptop webcam.
browser_camera_state = {"connected": False, "t": 0.0, "frame": None, "ws": None}
BROWSER_CAMERA_STALE_SECONDS = 2.0
LOCAL_CAMERA_INDEX = int(os.getenv("LOCAL_CAMERA_INDEX", "0"))
LOCAL_CAMERA_STALE_SECONDS = 2.0

# Server-owned absolute gimbal angles (degrees). Pi just applies them.
gimbal = {"pan": 90.0, "tilt": 90.0}


def pick_gimbal_target(detections):
    """Highest-confidence tracked drone. Birds/aircraft are never followed."""
    drones = [
        d for d in detections
        if d.get("track_id") is not None
        and str(d.get("class", "")).strip().lower() in DRONE_CLASSES
    ]
    return max(drones, key=lambda d: d["confidence"]) if drones else None


def update_gimbal(det, frame_w, frame_h):
    """Nudge pan/tilt toward the target. Returns True if the angles changed."""
    if det is None:
        return False
    turn_x = (det["cx"] - frame_w / 2.0) / (frame_w / 2.0)
    turn_y = (det["cy"] - frame_h / 2.0) / (frame_h / 2.0)
    moved = False
    if abs(turn_x) > GIMBAL_DEADZONE:
        step = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, turn_x * PAN_GAIN_DEG))
        gimbal["pan"] = max(0.0, min(180.0, gimbal["pan"] + PAN_SIGN * step))
        moved = True
    if abs(turn_y) > GIMBAL_DEADZONE:
        step = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, turn_y * TILT_GAIN_DEG))
        gimbal["tilt"] = max(0.0, min(180.0, gimbal["tilt"] + TILT_SIGN * step))
        moved = True
    return moved


async def send_gimbal(tracking):
    ws = pi_state["ws"]
    if ws is None:
        return
    try:
        await ws.send(json.dumps({
            "type": "gimbal",
            "pan": round(gimbal["pan"], 2),
            "tilt": round(gimbal["tilt"], 2),
            "tracking": bool(tracking),
            "timestamp": time.time(),
        }))
    except Exception as exc:
        print(f"Could not send gimbal command: {exc}")


class DetectionResults:
    """Minimal Ultralytics Results-like object accepted by BYTETracker."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape((-1, 4))
        self.conf = np.asarray(conf, dtype=np.float32).reshape((-1,))
        self.cls = np.asarray(cls, dtype=np.float32).reshape((-1,))

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, item):
        return DetectionResults(self.xywh[item], self.conf[item], self.cls[item])


def track_color(track_id):
    if track_id is None:
        return (200, 200, 200)
    track_id = int(track_id)
    if track_id not in TRACK_COLORS:
        hue = (track_id * 137.508) % 360.0
        hsv = np.uint8([[[hue / 2.0, 210.0, 255.0]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        TRACK_COLORS[track_id] = tuple(int(v) for v in bgr)
    return TRACK_COLORS[track_id]


def color_hex(bgr):
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


def choose_primary(detections):
    active = [d for d in detections if d.get("track_id") is not None]
    return max(active, key=lambda d: d["confidence"]) if active else None


def make_packet(detections, frame_width, frame_height, fps):
    return {
        "type": "yolo_detection",
        "timestamp": time.time(),
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "detected": bool(detections),
        "count": len(detections),
        "primary": choose_primary(detections),
        "detections": detections,
    }


def create_tracker():
    args = SimpleNamespace(
        track_high_thresh=TRACK_HIGH_THRESH,
        track_low_thresh=TRACK_LOW_THRESH,
        new_track_thresh=NEW_TRACK_THRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH,
        fuse_score=FUSE_SCORE,
        model="auto",
    )
    return BYTETracker(args)


def load_local_model():
    """Loads the trained model locally. Weights are downloaded once (needs the
    API key + internet), then cached on this machine."""
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. It is needed once to download the model weights."
        )
    print(f"Loading local model {LOCAL_MODEL_ID} ...")
    model = get_model(model_id=LOCAL_MODEL_ID, api_key=api_key)

    # Warm up, then time it so you know what frame rate to expect.
    dummy = np.zeros((480, 640, 3), np.uint8)
    model.infer(dummy, confidence=CONFIDENCE)
    runs = 8
    t0 = time.perf_counter()
    for _ in range(runs):
        model.infer(dummy, confidence=CONFIDENCE)
    ms = (time.perf_counter() - t0) / runs * 1000.0
    print(f"Local model ready: ~{ms:.0f} ms/frame (up to ~{1000.0 / max(ms, 1e-6):.1f} FPS)")
    return model


def _prediction_to_dict(p):
    """Local model prediction object -> same dict shape the old cloud path produced."""
    return {
        "x": p.x,
        "y": p.y,
        "width": p.width,
        "height": p.height,
        "confidence": p.confidence,
        "class_id": getattr(p, "class_id", 0),
        "class": getattr(p, "class_name", None) or "drone",
        "detection_id": getattr(p, "detection_id", None),
        "parent_id": None,
    }


def _run_local_model(model, frame):
    result = model.infer(frame, confidence=CONFIDENCE)
    result = result[0] if isinstance(result, list) else result
    return [_prediction_to_dict(p) for p in result.predictions]


def infer(client, tracker, frame):
    """Local model detection -> ByteTrack -> dashboard format."""
    raw_predictions = _run_local_model(client, frame)

    h, w = frame.shape[:2]
    detections_by_index = []
    xywh, conf, cls = [], [], []

    for prediction in raw_predictions:
        try:
            confidence = float(prediction.get("confidence", 0.0))
            if confidence < CONFIDENCE:
                continue

            cx = float(prediction["x"])
            cy = float(prediction["y"])
            box_w = float(prediction["width"])
            box_h = float(prediction["height"])

            class_id = int(prediction.get("class_id", 0))
            class_name = str(prediction.get("class", "drone"))

            x1 = max(0.0, cx - box_w / 2.0)
            y1 = max(0.0, cy - box_h / 2.0)
            x2 = min(float(w), cx + box_w / 2.0)
            y2 = min(float(h), cy + box_h / 2.0)

            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            if box_w <= 0.0 or box_h <= 0.0:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            detections_by_index.append(
                {
                    "class_id": class_id,
                    "class": class_name,
                    "confidence": confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy,
                    "width": box_w, "height": box_h,
                    "detection_id": prediction.get("detection_id"),
                    "parent_id": prediction.get("parent_id"),
                }
            )
            xywh.append([cx, cy, box_w, box_h])
            conf.append(confidence)
            cls.append(class_id)
        except (KeyError, TypeError, ValueError):
            continue

    results = DetectionResults(xywh, conf, cls)
    tracked = tracker.update(results, img=frame)

    output = []
    for row in tracked:
        if len(row) < 8:
            continue

        x1, y1, x2, y2 = map(float, row[:4])
        track_id = int(row[4])
        track_cls = int(row[6])
        detection_index = int(row[7])

        if detection_index < 0 or detection_index >= len(detections_by_index):
            continue

        source = detections_by_index[detection_index]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)

        output.append(
            {
                "track_id": track_id,
                "track_label": f"D-{track_id:02d}",
                "color_hex": color_hex(track_color(track_id)),
                "class_id": source["class_id"] if source["class_id"] is not None else track_cls,
                "class": source["class"],
                "confidence": source["confidence"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy,
                "width": box_w, "height": box_h,
            }
        )
    return output


async def send_to_clients(packet):
    if not clients:
        return
    message = json.dumps(packet, separators=(",", ":"))
    dead = set()
    for ws in list(clients):
        try:
            await ws.send(message)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


async def websocket_handler(websocket):
    """Dashboard + judge webcam connection.

    Text messages are ignored. Binary messages are JPEG frames sent by the
    browser camera. The newest browser frame becomes the preferred camera
    source for the inference pipeline.
    """
    clients.add(websocket)
    browser_camera_state["ws"] = websocket
    browser_camera_state["connected"] = True
    print(f"Dashboard connected: {len(clients)} client(s) from {websocket.remote_address}")
    try:
        async for message in websocket:
            if isinstance(message, (bytes, bytearray)):
                frame = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    browser_camera_state["frame"] = frame
                    browser_camera_state["t"] = time.time()
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.discard(websocket)
        if browser_camera_state["ws"] is websocket:
            browser_camera_state["ws"] = None
            browser_camera_state["connected"] = False
            browser_camera_state["t"] = 0.0
            browser_camera_state["frame"] = None
        print(f"Dashboard disconnected: {len(clients)} client(s)")


async def pi_handler(websocket):
    """Raspberry Pi connection: receives binary JPEG frames."""
    print(f"Raspberry Pi connected from {websocket.remote_address}")
    pi_state["connected"] = True
    pi_state["ws"] = websocket
    gimbal["pan"] = 90.0   # Pi centers its servos on startup; stay in sync
    gimbal["tilt"] = 90.0
    await send_gimbal(False)
    try:
        async for message in websocket:
            if not isinstance(message, (bytes, bytearray)):
                continue
            frame = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            pi_state["frame"] = frame
            pi_state["id"] += 1
            pi_state["t"] = time.time()
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        pi_state["connected"] = False
        pi_state["ws"] = None
        print("Raspberry Pi disconnected")


async def status_handler(websocket):
    """Dashboard connection for the Step 10 hardware-bridge panel."""
    status_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        status_clients.discard(websocket)


async def status_broadcaster():
    """Tells the dashboard which sensors are genuinely live (twice a second).

    Camera priority is Raspberry Pi first, then laptop webcam when the Pi is
    disconnected/offline. Radar/thermal/RF remain False in this setup.
    """
    while True:
        now = time.time()
        pi_live = bool(
            pi_state["connected"] and (now - pi_state["t"]) < PI_STALE_SECONDS
        )
        browser_live = bool(
            browser_camera_state["connected"]
            and browser_camera_state["frame"] is not None
            and (now - browser_camera_state["t"]) < BROWSER_CAMERA_STALE_SECONDS
        )
        laptop_live = bool(
            local_camera_state["connected"]
            and (now - local_camera_state["t"]) < LOCAL_CAMERA_STALE_SECONDS
        )
        camera_live = browser_live or pi_live or laptop_live
        camera_source = ("JUDGE_BROWSER" if browser_live else
                         ("RASPBERRY_PI" if pi_live else ("LAPTOP" if laptop_live else "NONE")))

        if status_clients:
            message = json.dumps({
                "type": "hardware_status",
                "camera": camera_live,
                "camera_source": camera_source,
                "radar": False,
                "thermal": False,
                "rf": False,
                "pi_connected": bool(pi_state["connected"]),
                "browser_camera": browser_live,
                "gimbal": {"pan": round(gimbal["pan"], 1), "tilt": round(gimbal["tilt"], 1)},
                "timestamp": now,
            }, separators=(",", ":"))
            dead = set()
            for ws in list(status_clients):
                try:
                    await ws.send(message)
                except Exception:
                    dead.add(ws)
            status_clients.difference_update(dead)
        await asyncio.sleep(0.5)


async def camera_loop(client, tracker):
    """
    Camera pipeline with automatic source failover.

    Judge browser camera has priority whenever a fresh browser frame is
    arriving. If no judge camera is connected, Raspberry Pi has priority,
    then the laptop webcam fallback. The YOLO/ByteTrack/dashboard packet
    format remains unchanged.
    """
    print(f"Dashboard WebSocket: ws://{HOST}:{PORT}")
    print(f"Pi WebSocket:        ws://<this-laptop-hotspot-ip>:{PI_PORT}")
    print(f"Local model: {LOCAL_MODEL_ID}")
    print("Camera source: judge browser webcam -> Raspberry Pi -> laptop webcam fallback.")

    latest_detections = []
    inference_fps = 0.0
    inference_alpha = 0.15
    inference_task = None
    last_inference_done = time.perf_counter()
    stream_counter = 0
    stream_window_start = time.perf_counter()
    stream_fps = 0.0

    last_pi_id = 0
    last_wait_msg = 0.0
    task_started = 0.0
    gimbal_last_move = 0.0

    local_cap = None
    local_retry_at = 0.0
    last_source = None

    async def collect_inference_task():
        nonlocal inference_task, latest_detections, inference_fps, last_inference_done
        if inference_task is None or not inference_task.done():
            return
        try:
            detections, elapsed = inference_task.result()
            latest_detections = detections
            instant = 1.0 / max(1e-6, elapsed)
            inference_fps = (
                instant if inference_fps <= 0.0
                else (1.0 - inference_alpha) * inference_fps + inference_alpha * instant
            )
            last_inference_done = time.perf_counter()
        except Exception as exc:
            print(f"Inference error: {exc}")
        finally:
            inference_task = None

    def run_inference(frame_for_inference):
        t0 = time.perf_counter()
        detections = infer(client, tracker, frame_for_inference)
        return detections, max(1e-6, time.perf_counter() - t0)

    def pi_is_fresh():
        return bool(
            pi_state["connected"]
            and pi_state["frame"] is not None
            and (time.time() - pi_state["t"]) < PI_STALE_SECONDS
        )

    def open_local_camera():
        nonlocal local_cap, local_retry_at
        if local_cap is not None and local_cap.isOpened():
            return True
        now = time.time()
        if now < local_retry_at:
            return False
        try:
            # CAP_DSHOW is more reliable for webcam startup on Windows.
            local_cap = cv2.VideoCapture(LOCAL_CAMERA_INDEX, cv2.CAP_DSHOW)
            if not local_cap.isOpened():
                local_cap.release()
                local_cap = None
                print(f"Could not open laptop webcam index {LOCAL_CAMERA_INDEX}; will retry.")
                local_retry_at = now + 2.0
                return False
            local_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            local_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
            local_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Laptop webcam opened (index {LOCAL_CAMERA_INDEX}).")
            return True
        except Exception as exc:
            local_cap = None
            local_retry_at = now + 2.0
            print(f"Laptop webcam open error: {exc}")
            return False

    def close_local_camera():
        nonlocal local_cap
        if local_cap is not None:
            try:
                local_cap.release()
            except Exception:
                pass
            local_cap = None

    try:
        while True:
            source = None
            frame = None

            # 1) Judge browser webcam has priority while fresh.
            browser_is_fresh = bool(
                browser_camera_state["connected"]
                and browser_camera_state["frame"] is not None
                and (time.time() - browser_camera_state["t"]) < BROWSER_CAMERA_STALE_SECONDS
            )
            if browser_is_fresh:
                source = "JUDGE_BROWSER"
                frame = browser_camera_state["frame"].copy()
                local_camera_state["connected"] = False
                local_camera_state["t"] = 0.0

            # 2) No browser camera -> Raspberry Pi has priority while live.
            if frame is None and pi_is_fresh():
                fid = pi_state["id"]
                if pi_state["frame"] is not None and fid != last_pi_id:
                    last_pi_id = fid
                    source = "RASPBERRY_PI"
                    frame = pi_state["frame"].copy()
                    # Pi is live, so the fallback camera is not considered active.
                    local_camera_state["connected"] = False
                    local_camera_state["t"] = 0.0

            # 3) No browser/Pi camera -> automatically use the laptop webcam.
            if frame is None:
                if open_local_camera():
                    ok, local_frame = local_cap.read()
                    if ok and local_frame is not None:
                        source = "LAPTOP"
                        frame = local_frame
                        local_camera_state["connected"] = True
                        local_camera_state["t"] = time.time()
                    else:
                        local_camera_state["connected"] = False
                        close_local_camera()
                else:
                    local_camera_state["connected"] = False

            # 4) Nothing available yet. Keep the WebSocket/status handlers alive.
            if frame is None:
                if time.time() - last_wait_msg > 3.0:
                    if browser_camera_state["connected"]:
                        print("Judge browser camera is stale; waiting for a fresh frame...")
                    elif pi_state["connected"]:
                        print("Pi stream is stale; waiting for laptop webcam fallback...")
                    else:
                        print("No judge/Pi stream; waiting for laptop webcam...")
                    last_wait_msg = time.time()
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                await asyncio.sleep(0.01)
                continue

            if source != last_source:
                print(f"ACTIVE CAMERA SOURCE -> {source}")
                last_source = source

            prev_done = last_inference_done
            await collect_inference_task()

            if last_inference_done != prev_done:
                target = pick_gimbal_target(latest_detections)
                if task_started <= gimbal_last_move + GIMBAL_SETTLE_SECONDS:
                    print("[gimbal] result skipped: frame captured before servos settled")
                elif target is None:
                    print(f"[gimbal] no drone target; tracks={[d['class'] for d in latest_detections]}")
                elif update_gimbal(target, frame.shape[1], frame.shape[0]):
                    gimbal_last_move = time.time()
                    # send_gimbal() is a no-op when no Pi is connected.
                    await send_gimbal(True)
                    print(
                        f"[gimbal] -> pan={gimbal['pan']:.1f} tilt={gimbal['tilt']:.1f} "
                        f"(Pi connected: {pi_state['connected']})"
                    )
                else:
                    print("[gimbal] target inside deadzone, no move needed")

            if inference_task is None:
                task_started = time.time()
                inference_task = asyncio.create_task(
                    asyncio.to_thread(run_inference, frame.copy())
                )

            stream_counter += 1
            now_perf = time.perf_counter()
            window_elapsed = now_perf - stream_window_start
            if window_elapsed >= 0.5:
                stream_fps = stream_counter / window_elapsed
                stream_counter = 0
                stream_window_start = now_perf

            h, w = frame.shape[:2]

            for d in latest_detections:
                x1, y1 = int(d["x1"]), int(d["y1"])
                x2, y2 = int(d["x2"]), int(d["y2"])
                label = f'{d["track_label"]} | {d["class"]} {d["confidence"]:.2f}'
                color = track_color(d.get("track_id"))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            packet = make_packet(latest_detections, w, h, stream_fps)
            packet["stream_fps"] = stream_fps
            packet["inference_fps"] = inference_fps
            packet["inference_age_ms"] = (time.perf_counter() - last_inference_done) * 1000.0
            packet["camera_source"] = source
            packet["pi_connected"] = bool(pi_state["connected"])
            packet["browser_camera"] = bool(browser_camera_state["connected"])

            encode_ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            packet["frame_jpeg_b64"] = (
                base64.b64encode(encoded).decode("ascii") if encode_ok else None
            )

            await send_to_clients(packet)

            if os.getenv("SHOW_LOCAL_PREVIEW", "0") == "1":
                cv2.putText(
                    frame,
                    f"{source} | {stream_fps:.1f} FPS | Model: {inference_fps:.1f} FPS | tracks: {len(latest_detections)}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 0), 2,
                )
                cv2.imshow("Camera feed -> Dashboard", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            await asyncio.sleep(0)
    finally:
        close_local_camera()
        local_camera_state["connected"] = False
        local_camera_state["t"] = 0.0
        browser_camera_state["connected"] = False
        browser_camera_state["t"] = 0.0
        browser_camera_state["frame"] = None
        if os.getenv("SHOW_LOCAL_PREVIEW", "0") == "1":
            cv2.destroyAllWindows()


async def main():
    client = load_local_model()
    tracker = create_tracker()

    async with websockets.serve(websocket_handler, HOST, PORT, max_size=4 * 1024 * 1024), \
               websockets.serve(status_handler, HOST, STATUS_PORT), \
               websockets.serve(pi_handler, PI_HOST, PI_PORT, max_size=4 * 1024 * 1024, compression=None):
        status_task = asyncio.create_task(status_broadcaster())
        try:
            await camera_loop(client, tracker)
        finally:
            status_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")