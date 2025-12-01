import os
import time
import csv
from datetime import datetime
import math
import threading
import smtplib
import ssl
from email.message import EmailMessage
import tkinter as tk
from tkinter import simpledialog
import cv2
import numpy as np
import sys
import urllib.request

# Optional libs (graceful fallback)
try:
    import mediapipe as mp
except Exception as e:
    print("ERROR: mediapipe is required. Install: pip install mediapipe")
    raise

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    print("ultralytics not installed. YOLO features disabled. Install: pip install ultralytics")

try:
    import pyttsx3
except Exception:
    pyttsx3 = None
    print("pyttsx3 not installed (TTS disabled).")

try:
    import sounddevice as sd
except Exception:
    sd = None
    print("sounddevice not installed (audio RMS disabled).")

try:
    from fpdf import FPDF
except Exception:
    FPDF = None
    print("fpdf2 not installed (PDF generation disabled).")

# ----------------------------- SETTINGS -----------------------------
AUTO_DOWNLOAD_YOLO = True        # you selected "A" — attempts to download yolov8n.pt if missing
YOLO_FILENAME = "yolov8n.pt"     # model file used by ultralytics YOLO
YOLO_DOWNLOAD_CANDIDATES = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "https://ultralytics.com/models/yolov8n.pt",
    "https://github.com/ultralytics/ultralytics/releases/latest/download/yolov8n.pt"
]

# Email credentials (replace with your own or prompt)
TEACHER_EMAIL = "abc123@gmail.com"
SENDER_EMAIL = "abc123@gmail.com"
APP_PASSWORD = "Anil123#12"   # Gmail app password recommended

# Exit password:
EXIT_PASSWORD = "exit@123"

# ----------------------------- USER METADATA -----------------------------
candidate_name = input("Enter candidate name: ").strip() or "Unknown"
candidate_id = input("Enter candidate id/roll no (optional): ").strip()

# ----------------------------- FILES & FOLDERS -----------------------------
TIMESTAMP = int(time.time())
VIDEO_OUTPUT = f"session_record_{candidate_name}_{TIMESTAMP}.mp4"
CSV_LOG = f"session_log_{candidate_name}_{TIMESTAMP}.csv"
PDF_REPORT = f"session_report_{candidate_name}_{TIMESTAMP}.pdf"
SCREENSHOT_DIR = f"screenshots_{candidate_name}_{TIMESTAMP}"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

KNOWN_FACES_DIR = "known_faces"  # optional: for LBPH

# ----------------------------- THRESHOLDS -----------------------------
MAX_YAW_OFFSET = 30.0
MAX_PITCH_OFFSET = 35.0
HAND_NEAR_FACE_DISTANCE = 0.15
HAND_MOVEMENT_VELOCITY_THRESHOLD = 0.02
AUDIO_RMS_THRESHOLD = 0.02
AUDIO_CHECK_INTERVAL = 0.5

# ----------------------------- GLOBAL STATE -----------------------------
away_count = 0
phone_detected_count = 0
unauth_person_count = 0
total_frames = 0
events = []  # (timestamp, event, extra, screenshot_path)
session_start = time.time()
audio_rms = 0.0
last_event_screenshot = None
last_audio_alert_time = 0
prev_hand_positions = []

cheating_score = 0
EVENT_SCORE_MAP = {
    "Looking Away": 5,
    "Phone Detected": 20,
    "Multiple Persons Detected": 30,
    "Hand Near Face": 7,
    "Rapid Hand Movement": 8,
    "Suspicious Audio": 10,
    "Unknown face detected": 15,
    "Unauthorized Person (face mismatch)": 25,
    "Cheat Sheet / Paper": 20,
    "Unauthorized Paper": 20,
    "Book Detected": 20,
    "Smartwatch Detected": 15,
    "Earphone Detected": 15,
    "Calculator Detected": 10,
    "Tablet Detected": 15,
}

# ----------------------------- TTS -----------------------------
def init_tts():
    if not pyttsx3:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        return engine
    except Exception:
        return None

tts_engine = init_tts()

def speak(text):
    if tts_engine:
        try:
            tts_engine.say(text)
            tts_engine.iterate()
        except Exception:
            print("[AUDIO]", text)
    else:
        print("[AUDIO]", text)

# ----------------------------- UTILITIES -----------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def increase_score_for(event_name):
    global cheating_score
    inc = EVENT_SCORE_MAP.get(event_name, 0)
    if inc:
        cheating_score += inc
        cheating_score = min(cheating_score, 100)

def log_event(event_type, extra="", frame=None):
    global last_event_screenshot
    ts = now_str()
    shot_path = ""
    if frame is not None:
        shot_path = os.path.join(SCREENSHOT_DIR, f"{event_type.replace(' ','')}_{int(time.time())}.jpg")
        try:
            small = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))
            cv2.imwrite(shot_path, small)
        except Exception:
            shot_path = ""
    events.append((ts, event_type, extra, shot_path))
    append_to_csv([ts, event_type, extra, shot_path])
    last_event_screenshot = shot_path
    increase_score_for(event_type)
    print(f"[EVENT] {ts} | {event_type} | {extra}")

# CSV utilities
with open(CSV_LOG, mode="w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["timestamp","event","extra","screenshot"])

def append_to_csv(row):
    with open(CSV_LOG, mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)

# ----------------------------- LBPH FACE RECOGNITION (optional) -----------------------------
use_face_recognition = False
face_recognizer = None
label_map = {}

def prepare_lbph_known_faces():
    global face_recognizer, label_map, use_face_recognition
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception:
        return
    data, labels = [], []
    label_id = 0
    label_map = {}
    if not os.path.exists(KNOWN_FACES_DIR):
        return
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    for name in os.listdir(KNOWN_FACES_DIR):
        sub = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(sub): continue
        label_map[label_id] = name
        for imgf in os.listdir(sub):
            p = os.path.join(sub, imgf)
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None: continue
            rects = face_cascade.detectMultiScale(im, scaleFactor=1.1, minNeighbors=4)
            if len(rects) == 0:
                im2 = cv2.resize(im, (200,200))
                data.append(im2); labels.append(label_id)
            else:
                x,y,w,h = rects[0]
                face = cv2.resize(im[y:y+h,x:x+w], (200,200))
                data.append(face); labels.append(label_id)
        label_id += 1
    if len(data) >= 2:
        recognizer.train(data, np.array(labels))
        face_recognizer = recognizer
        use_face_recognition = True
        print("LBPH trained:", label_map)

prepare_lbph_known_faces()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------- YOLO MODEL (auto-download if requested) -----------------------------
def ensure_yolo_model(path=YOLO_FILENAME):
    """Ensure YOLO model file exists; attempt downloads from candidate urls."""
    if os.path.exists(path):
        print(f"YOLO model found: {path}")
        return True
    if not AUTO_DOWNLOAD_YOLO or YOLO is None:
        print(f"YOLO model missing ({path}). Set AUTO_DOWNLOAD_YOLO=True or download manually.")
        return False
    print(f"YOLO model {path} not found — attempting download...")
    for url in YOLO_DOWNLOAD_CANDIDATES:
        try:
            print("Trying:", url)
            urllib.request.urlretrieve(url, path)
            if os.path.exists(path) and os.path.getsize(path) > 100_000:  # basic size check
                print("Downloaded YOLO model to", path)
                return True
            else:
                print("Downloaded file appears too small; removing.")
                try: os.remove(path)
                except: pass
        except Exception as e:
            print("Download attempt failed:", e)
    print("All download attempts failed. Please download yolov8n.pt manually and place in the script folder.")
    return False

yolo_model = None
if YOLO is not None:
    if ensure_yolo_model(YOLO_FILENAME):
        try:
            yolo_model = YOLO(YOLO_FILENAME)
            print("Loaded YOLO model:", YOLO_FILENAME)
        except Exception as e:
            print("Failed to load YOLO model:", e)
            yolo_model = None
    else:
        yolo_model = None
else:
    yolo_model = None

# Robust cheating object mapping using substring checks
CHEATING_KEYWORDS = {
    "phone": "Phone Detected",
    "cell phone": "Phone Detected",
    "smartphone": "Phone Detected",
    "paper": "Cheat Sheet / Paper",
    "document": "Unauthorized Paper",
    "book": "Book Detected",
    "notebook": "Cheat Sheet / Paper",
    "laptop": "Laptop Detected",
    "keyboard": "Laptop Usage",
    "monitor": "External Screen",
    "tv": "External Screen",
    "tablet": "Tablet Detected",
    "watch": "Smartwatch Detected",
    "smartwatch": "Smartwatch Detected",
    "earphone": "Earphone Detected",
    "earphones": "Earphone Detected",
    "earbud": "Earphone Detected",
    "headphones": "Earphone Detected",
    "headset": "Earphone Detected",
    "remote": "Hidden Device Detected",
    "mouse": "Computer Interaction Detected",
    "calculator": "Calculator Detected",
    "bag": "Suspicious Object",
    "book": "Book Detected"
}

def map_yolo_class_to_event(cls_name):
    """Return mapped event name for a YOLO class name using substring matching."""
    if not cls_name:
        return None
    cls = cls_name.lower()
    for k, v in CHEATING_KEYWORDS.items():
        if k in cls:
            return v
    return None

# ----------------------------- MEDIAPIPE & AUDIO THREAD -----------------------------
mp_face_mesh_class = mp.solutions.face_mesh.FaceMesh
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False,
                                max_num_hands=2,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.4)

# head-pose model points
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

def estimate_head_pose(landmarks, width, height):
    try:
        image_points = np.array([
            (landmarks[1].x * width, landmarks[1].y * height),
            (landmarks[152].x * width, landmarks[152].y * height),
            (landmarks[33].x * width, landmarks[33].y * height),
            (landmarks[263].x * width, landmarks[263].y * height),
            (landmarks[61].x * width, landmarks[61].y * height),
            (landmarks[291].x * width, landmarks[291].y * height)
        ], dtype=np.float64)
        focal_length = width
        camera_matrix = np.array([[focal_length, 0, width/2], [0, focal_length, height/2], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4,1))
        success, rot_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            return None
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        return angles
    except Exception:
        return None

# Audio RMS monitor (daemon thread)
audio_running = True
def audio_monitor_thread():
    global audio_rms, audio_running
    if sd is None:
        return
    try:
        samplerate = 16000
        blocksize = int(samplerate * AUDIO_CHECK_INTERVAL)
        def callback(indata, frames, time_info, status):
            global audio_rms
            if status:
                pass
            rms = np.sqrt(np.mean(indata.astype(np.float32)**2))
            audio_rms = float(rms)
        with sd.InputStream(channels=1, samplerate=samplerate, blocksize=blocksize, callback=callback):
            while audio_running:
                time.sleep(AUDIO_CHECK_INTERVAL)
    except Exception as e:
        print("Audio monitor error:", e)
        return

if sd is not None:
    t_audio = threading.Thread(target=audio_monitor_thread, daemon=True)
    t_audio.start()

# ----------------------------- VIDEO CAPTURE SETUP -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (frame_w, frame_h))

# Pre-calibration countdown
window_name = "Proctoring System"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
for i in range(5,0,-1):
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, "Face camera straight for calibration", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Starting in {i}s", (30,140), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 3)
    cv2.imshow(window_name, frame)
    cv2.waitKey(1000)

# Calibration
calib_list = []
calib_start = time.time()
with mp_face_mesh_class(refine_landmarks=True) as fm:
    while time.time() - calib_start < 3:
        ret, frame = cap.read()
        if not ret:
            break
        h,w,_ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            angles = estimate_head_pose(lm, w, h)
            if angles:
                p,y,_ = angles
                calib_list.append((p,y))
        cv2.putText(frame, "Calibrating... Keep your face straight", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if calib_list:
    calibrated_pitch = np.mean([p for p,y in calib_list])
    calibrated_yaw = np.mean([y for p,y in calib_list])
else:
    calibrated_pitch = 0.0
    calibrated_yaw = 0.0

speak("Calibration complete. Monitoring started.")

# Anti-spoof blink/motion helpers
blink_count = 0
blink_last_state = False
last_face_area = None
spoof_suspect_since = None

def detect_blink_and_motion(face_landmarks):
    global blink_count, blink_last_state, last_face_area
    try:
        L_TOP, L_BOTTOM = 159, 145
        R_TOP, R_BOTTOM = 386, 374
        l_dist = abs(face_landmarks[L_TOP].y - face_landmarks[L_BOTTOM].y)
        r_dist = abs(face_landmarks[R_TOP].y - face_landmarks[R_BOTTOM].y)
        eye_avg = (l_dist + r_dist) / 2.0
        BLINK_THRESH = 0.01
        blink_now = eye_avg < BLINK_THRESH
    except Exception:
        blink_now = False
    try:
        xs = [p.x for idx,p in enumerate(face_landmarks) if idx % 10 == 0][:20]
        ys = [p.y for idx,p in enumerate(face_landmarks) if idx % 10 == 0][:20]
        if xs and ys:
            area = (max(xs)-min(xs))*(max(ys)-min(ys))
        else:
            area = None
    except Exception:
        area = None
    if blink_now and not blink_last_state:
        blink_count += 1
    blink_last_state = blink_now
    motion = False
    if last_face_area is not None and area is not None:
        if abs(area - last_face_area) > 0.0008:
            motion = True
    if area is not None:
        last_face_area = area
    return blink_now, blink_count, motion

# ----------------------------- PASSWORD-EXIT DIALOG -----------------------------
def ask_exit_password():
    root = tk.Tk()
    root.withdraw()
    try:
        answer = simpledialog.askstring("Exit Password", "Enter exit password to terminate session:", show='*')
        root.destroy()
        if answer is None:
            return False
        return answer == EXIT_PASSWORD
    except Exception:
        try:
            root.destroy()
        except:
            pass
        return False

# ----------------------------- MAIN LOOP -----------------------------
with mp_face_mesh_class(refine_landmarks=True) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        h,w,_ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mesh_res = face_mesh.process(rgb)
        hands_res = hands_detector.process(rgb)

        # YOLO detection (best-effort)
        yres = None
        if yolo_model is not None:
            try:
                results = yolo_model(frame)  # detection
                if len(results) > 0:
                    yres = results[0]
            except Exception as e:
                # graceful degrade
                # print("YOLO inference error:", e)
                yres = None

        # overlays background
        cv2.rectangle(frame, (0,0), (480,160), (30,30,30), -1)
        cv2.putText(frame, f"Candidate: {candidate_name}", (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"ID: {candidate_id}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # YOLO object mapping (robust substring)
        if yres is not None and hasattr(yres, "boxes") and hasattr(yres, "names"):
            try:
                for box in yres.boxes:
                    cls_index = int(box.cls[0])
                    cls_name = yres.names.get(cls_index, str(cls_index)).lower() if isinstance(yres.names, dict) else str(yres.names[cls_index]).lower()
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    mapped = map_yolo_class_to_event(cls_name)
                    if mapped:
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                        cv2.putText(frame, mapped, (x1, max(y1-6,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        # only log once per event occurrence (but we keep simple: log with frame)
                        log_event(mapped, extra=f"class={cls_name}", frame=frame)
                        # audio rate-limit
                        if time.time() - last_audio_alert_time > 2:
                            speak(mapped)
                            last_audio_alert_time = time.time()
                        if "phone" in cls_name or "cell" in cls_name:
                            phone_detected_count += 1
            except Exception as e:
                # ignore YOLO mapping errors
                pass

        # Person count detection (if YOLO provides person class)
        person_count = 0
        if yres is not None and hasattr(yres, "boxes"):
            try:
                names = yres.names
                person_count = sum(1 for box in yres.boxes if ("person" in (names[int(box.cls[0])].lower() if isinstance(names, dict) else str(names[int(box.cls[0])]).lower())))
            except Exception:
                person_count = 0
        if person_count > 1:
            unauth_person_count += 1
            log_event("Multiple Persons Detected", extra=f"count={person_count}", frame=frame)
            if time.time() - last_audio_alert_time > 4:
                speak("Multiple people detected")
                last_audio_alert_time = time.time()
            cv2.putText(frame, "MULTIPLE PEOPLE!", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Head pose & looking away
        pitch = 0.0; yaw = 0.0
        if mesh_res.multi_face_landmarks:
            lm = mesh_res.multi_face_landmarks[0].landmark
            angles = estimate_head_pose(lm, w, h)
            if angles:
                pitch, yaw, _ = angles
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (300,26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (300,46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1)
                if (abs(pitch - calibrated_pitch) > MAX_PITCH_OFFSET) or (abs(yaw - calibrated_yaw) > MAX_YAW_OFFSET):
                    away_count += 1
                    log_event("Looking Away", extra=f"pitch={pitch:.1f},yaw={yaw:.1f}", frame=frame)
                    if time.time() - last_audio_alert_time > 2:
                        speak("Looking away detected")
                        last_audio_alert_time = time.time()
                    cv2.putText(frame, "LOOKING AWAY!", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # anti-spoof
            blink_now, bcount, motion = detect_blink_and_motion(lm)
            if not motion and bcount < 1:
                if 'spoof_suspect_since' not in globals() or spoof_suspect_since is None:
                    spoof_suspect_since = time.time()
                elif time.time() - spoof_suspect_since > 12:
                    log_event("AntiSpoof: No blink/no motion (suspect)", frame=frame)
                    speak("Possible spoof detected")
            else:
                spoof_suspect_since = None

        # Hands: positions, near face, velocity
        hand_positions = []
        if hands_res.multi_hand_landmarks:
            for hand_landmarks in hands_res.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                hand_positions.append((wrist.x, wrist.y))
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)
                if mesh_res.multi_face_landmarks:
                    nose = mesh_res.multi_face_landmarks[0].landmark[1]
                    dx = wrist.x - nose.x
                    dy = wrist.y - nose.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < HAND_NEAR_FACE_DISTANCE:
                        log_event("Hand Near Face", extra=f"dist={dist:.3f}", frame=frame)
                        if time.time() - last_audio_alert_time > 2:
                            speak("Hand near face detected")
                            last_audio_alert_time = time.time()
                        cv2.putText(frame, "HAND NEAR FACE", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # hand movement velocity
        if prev_hand_positions and hand_positions:
            vels = []
            for i,pos in enumerate(hand_positions):
                prev = prev_hand_positions[min(i, len(prev_hand_positions)-1)]
                vx = abs(pos[0] - prev[0])
                vy = abs(pos[1] - prev[1])
                v = math.sqrt(vx*vx + vy*vy)
                vels.append(v)
            if vels and max(vels) > HAND_MOVEMENT_VELOCITY_THRESHOLD:
                log_event("Rapid Hand Movement", extra=f"vel={max(vels):.4f}", frame=frame)
                if time.time() - last_audio_alert_time > 2:
                    speak("Rapid hand movement detected")
                    last_audio_alert_time = time.time()
        prev_hand_positions = hand_positions.copy()

        # Face recognition LBPH (optional)
        if use_face_recognition and face_recognizer is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                if len(rects) > 0:
                    rects = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
                    x,y,wf,hf = rects[0]
                    face_img = cv2.resize(gray[y:y+hf, x:x+wf], (200,200))
                    label_id, conf = face_recognizer.predict(face_img)
                    if conf < 70:
                        name = label_map.get(label_id, "Unknown")
                        cv2.putText(frame, f"{name} ({conf:.0f})", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        if name != candidate_name:
                            log_event("Unauthorized Person (face mismatch)", extra=f"detected={name},conf={conf:.1f}", frame=frame)
                            if time.time() - last_audio_alert_time > 3:
                                speak("Unauthorized person detected")
                                last_audio_alert_time = time.time()
                    else:
                        cv2.putText(frame, f"Unknown ({conf:.0f})", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        log_event("Unknown face detected", extra=f"conf={conf:.1f}", frame=frame)
                        if time.time() - last_audio_alert_time > 3:
                            speak("Unknown person detected")
                            last_audio_alert_time = time.time()
            except Exception:
                pass

        # audio whisper detection
        if sd is not None:
            if audio_rms > AUDIO_RMS_THRESHOLD:
                log_event("Suspicious Audio", extra=f"rms={audio_rms:.4f}", frame=frame)
                if time.time() - last_audio_alert_time > 3:
                    speak("Suspicious audio detected")
                    last_audio_alert_time = time.time()

        # ========================= REAL-TIME STATUS (priority-based) =========================
        status = "OK"; color = (0,255,0)
        looking_away_now = False
        hand_near_face_now = False
        multi_person_now = (person_count > 1)
        phone_now = False

        try:
            if mesh_res.multi_face_landmarks:
                if (abs(pitch - calibrated_pitch) > MAX_PITCH_OFFSET) or (abs(yaw - calibrated_yaw) > MAX_YAW_OFFSET):
                    looking_away_now = True
        except Exception:
            looking_away_now = False

        if yres is not None and hasattr(yres, "boxes") and hasattr(yres, "names"):
            try:
                for box in yres.boxes:
                    cls_index = int(box.cls[0])
                    cls_name = yres.names.get(cls_index, str(cls_index)).lower() if isinstance(yres.names, dict) else str(yres.names[cls_index]).lower()
                    if "phone" in cls_name or "cell" in cls_name:
                        phone_now = True
                        break
            except Exception:
                pass

        if hands_res.multi_hand_landmarks and mesh_res.multi_face_landmarks:
            for hand_landmarks in hands_res.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                nose = mesh_res.multi_face_landmarks[0].landmark[1]
                dx = wrist.x - nose.x
                dy = wrist.y - nose.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < HAND_NEAR_FACE_DISTANCE:
                    hand_near_face_now = True
                    break

        # priority: multiple persons / phone => CHEATING, else looking-away/hand => SUSPICIOUS, else OK
        if multi_person_now:
            status = "CHEATING"; color = (0,0,255)
        elif phone_now:
            status = "CHEATING"; color = (0,0,255)
        elif looking_away_now:
            status = "SUSPICIOUS"; color = (0,165,255)
        elif hand_near_face_now:
            status = "SUSPICIOUS"; color = (0,165,255)
        else:
            status = "OK"; color = (0,255,0)

        cv2.putText(frame, f"STATUS: {status}", (320,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # overlays
        elapsed = int(time.time() - session_start)
        looking_pct = (away_count / total_frames * 100) if total_frames else 0.0
        cv2.putText(frame, f"Time: {elapsed}s", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(frame, f"Score: {cheating_score}", (10,95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"LookingAway: {looking_pct:.1f}%", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # write & show
        out.write(frame)
        cv2.imshow(window_name, frame)

        # exit handler
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            ok = ask_exit_password()
            if ok:
                print("Exit password correct — ending session.")
                break
            else:
                print("Incorrect exit password — continuing session.")
                speak("Incorrect password. Session continues.")
                continue

# ----------------------------- CLEANUP -----------------------------
audio_running = False
try:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
except Exception:
    pass

# Final summary stats
duration = int(time.time() - session_start)
looking_pct = (away_count / total_frames * 100) if total_frames else 0.0
phone_pct = (phone_detected_count / total_frames * 100) if total_frames else 0.0
multi_pct = (unauth_person_count / total_frames * 100) if total_frames else 0.0

append_to_csv([now_str(), "SESSION_SUMMARY", f"duration_s={duration},frames={total_frames},score={cheating_score}", ""])

# ----------------------------- PDF REPORT -----------------------------
if FPDF:
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, "Proctoring Session Report", ln=1, align="C")
        pdf.ln(4)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 6, f"Candidate: {candidate_name}", ln=1)
        if candidate_id: pdf.cell(0,6, f"ID: {candidate_id}", ln=1)
        pdf.cell(0, 6, f"Session Duration (s): {duration}", ln=1)
        pdf.cell(0, 6, f"Total Frames: {total_frames}", ln=1)
        pdf.cell(0, 6, f"Cheating Score: {cheating_score}", ln=1)
        pdf.cell(0, 6, f"Looking Away: {looking_pct:.2f}%", ln=1)
        pdf.cell(0, 6, f"Phone Detection: {phone_pct:.2f}%", ln=1)
        pdf.cell(0, 6, f"Multiple People Detection: {multi_pct:.2f}%", ln=1)
        pdf.ln(6)
        pdf.cell(0, 6, "Event Log (last 50):", ln=1)
        pdf.set_font("Arial", size=9)
        for ts, evt, extra, shot in events[-50:]:
            pdf.multi_cell(0, 5, f"{ts}  |  {evt}  |  {extra}")
            if shot and os.path.exists(shot):
                try:
                    pdf.image(shot, w=120)
                    pdf.ln(2)
                except Exception:
                    pass
        pdf.output(PDF_REPORT)
        print("PDF saved to", PDF_REPORT)
    except Exception as e:
        print("PDF generation failed:", e)
else:
    print("FPDF not installed — skipping PDF generation.")

# ----------------------------- EMAIL REPORT -----------------------------
def send_email_with_attachments(sender_email, app_password, receiver_email, subject, body, files):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg.set_content(body)
        for path in files:
            if not path or not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                data = f.read()
            maintype = "application"
            subtype = "octet-stream"
            if path.lower().endswith(".pdf"):
                maintype, subtype = "application", "pdf"
            elif path.lower().endswith(".csv"):
                maintype, subtype = "text", "csv"
            elif path.lower().endswith((".jpg", ".jpeg", ".png")):
                maintype, subtype = "image", os.path.splitext(path)[1].replace(".", "")
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(path))
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        return True, ""
    except Exception as e:
        return False, str(e)

attachments = [PDF_REPORT, CSV_LOG]
last_shots = [shot for (ts, evt, extra, shot) in events if shot]
if last_shots:
    attachments += last_shots[-3:]

subject = f"Proctoring Report: {candidate_name} ({now_str()})"
body = f"Proctoring session finished for {candidate_name} (ID: {candidate_id}). Cheating score: {cheating_score}.\n\nSee attachments."

print("Sending email to teacher...")
ok, err = send_email_with_attachments(SENDER_EMAIL, APP_PASSWORD, TEACHER_EMAIL, subject, body, attachments)
if ok:
    print("Email sent to", TEACHER_EMAIL)
else:
    print("Email failed:", err)

# final console summary
print("\n=== SESSION SUMMARY ===")
print(f"Candidate: {candidate_name}  ID: {candidate_id}")
print(f"Duration: {duration} seconds")
print(f"Total frames: {total_frames}")
print(f"Events logged: {len(events)}")
print(f"Cheating Score: {cheating_score}")
print(f"Video saved to: {VIDEO_OUTPUT}")
print(f"CSV log saved to: {CSV_LOG}")
if FPDF:
    print(f"PDF report saved to: {PDF_REPORT}")

speak("Session finished. Report generated and emailed.")
