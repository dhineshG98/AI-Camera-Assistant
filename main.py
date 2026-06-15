from ultralytics import YOLO
import cv2
import threading
import time
from collections import defaultdict
import tkinter as tk
from queue import Queue, Empty
from vosk import Model as VoskModel, KaldiRecognizer
import pyaudio, json
import asyncio
import edge_tts
import pygame
import os
import uuid
import numpy as np

# ------------------ CONFIG ------------------
MIN_CONFIDENCE = 0.45
MEMORY_FRAMES_REQUIRED = 2

# Priority only controls speaking order, NOTHING is hidden
OBJECT_PRIORITY = {
    "person": 5,
    "cell phone": 5,
    "bottle": 4,
    "headphones": 4,
    "car": 4,
    "bus": 4,
    "truck": 4,
    "motorcycle": 3,
    "bicycle": 3,
    "chair": 2,
    "table": 2,
    "bed": 2
}

# ------------------ MODELS ------------------
model = YOLO("yolov8m.pt")
vosk_model = VoskModel(r"C:\Users\rlath\OneDrive\Desktop\vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(vosk_model, 16000)

# ------------------ SPEECH SYSTEM ------------------
speech_queue = Queue()

def clear_speech_queue():
    """Remove any pending old speech so only latest is spoken"""
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except Empty:
            break

def speak(text):
    if text:
        speech_queue.put(text)

def speak_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pygame.mixer.init()
    pygame.mixer.music.set_volume(1.0)

    while True:
        text = speech_queue.get()
        audio_file = f"voice_{uuid.uuid4().hex}.mp3"

        try:
            pygame.mixer.music.stop()  # interrupt old speech immediately

            loop.run_until_complete(
                edge_tts.Communicate(text, voice="en-US-AriaNeural").save(audio_file)
            )

            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(20)

            pygame.mixer.music.unload()
            os.remove(audio_file)

        except Exception as e:
            print("Speech error:", e)

        speech_queue.task_done()

threading.Thread(target=speak_worker, daemon=True).start()

# ------------------ VOICE COMMANDS ------------------
command_queue = Queue()

def listen_command():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=8000
    )
    stream.start_stream()

    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            cmd = json.loads(rec.Result()).get("text", "").lower().strip()
            if cmd:
                command_queue.put(cmd)

# ------------------ CAMERA + DETECTION ------------------
def run_camera(status_label, logbox, root):
    detection_active = False
    object_memory = defaultdict(int)
    last_scene_signature = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        scene_objects = defaultdict(int)

        # -------- VOICE COMMANDS --------
        try:
            while True:
                cmd = command_queue.get_nowait()

                if "start" in cmd:
                    detection_active = True
                    status_label.config(text="Detection Active")

                elif "stop detection" in cmd or "stop" in cmd:
                    detection_active = False
                    status_label.config(text="Detection Paused")

                elif "stop camera" in cmd:
                    cap.release()
                    cv2.destroyAllWindows()
                    root.destroy()
                    return
        except Empty:
            pass

        # -------- YOLO DETECTION --------
        if detection_active:
            small = cv2.resize(frame, (320, 240))
            results = model.predict(small, imgsz=320, conf=MIN_CONFIDENCE, verbose=False)

            for r in results:
                for b in r.boxes:
                    if float(b.conf[0]) < MIN_CONFIDENCE:
                        continue

                    label = model.names[int(b.cls[0])]

                    x1, y1, x2, y2 = b.xyxy[0]
                    x1 = int(x1 * (w / 320))
                    x2 = int(x2 * (w / 320))
                    y1 = int(y1 * (h / 240))
                    y2 = int(y2 * (h / 240))

                    cx = (x1 + x2) // 2
                    region = (
                        "left" if cx < w / 3 else
                        "right" if cx > 2 * w / 3 else
                        "center"
                    )

                    key = (label, region)
                    object_memory[key] += 1

                    if object_memory[key] >= MEMORY_FRAMES_REQUIRED:
                        scene_objects[key] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} ({region})",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        # -------- SCENE-AWARE SPEECH (SYNCED WITH GUI) --------
        if scene_objects:
            signature = tuple(sorted(scene_objects.items()))

            if signature != last_scene_signature:
                items = []
                for (obj, region), count in scene_objects.items():
                    priority = OBJECT_PRIORITY.get(obj, 1)
                    items.append((priority, obj, region, count))

                items.sort(reverse=True)

                parts = [
                    f"{count} {obj}{'s' if count > 1 else ''} on the {region}"
                    for _, obj, region, count in items
                ]

                spoken_text = "I see " + ", ".join(parts)

                clear_speech_queue()   # 🔥 critical fix
                speak(spoken_text)
                logbox.insert(tk.END, spoken_text + "\n")
                logbox.see(tk.END)

                last_scene_signature = signature
                object_memory.clear()

        cv2.imshow("AI Camera Assistant", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ WARMUP (REMOVES START LAG) ------------------
def warmup_system():
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=320, conf=0.3, verbose=False)
    speak(" ")

# ------------------ GUI ------------------
root = tk.Tk()
root.title("AI Camera Assistant")
root.geometry("520x420")

status_label = tk.Label(root, text="Detection OFF", font=("Arial", 16))
status_label.pack(pady=10)

scrollbar = tk.Scrollbar(root)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

logbox = tk.Text(root, height=15, width=60, yscrollcommand=scrollbar.set)
logbox.pack(pady=10)
scrollbar.config(command=logbox.yview)

threading.Thread(target=listen_command, daemon=True).start()
threading.Thread(target=run_camera, args=(status_label, logbox, root), daemon=True).start()

warmup_system()
speak("Voice system ready. Say start to begin detection.")

root.mainloop()
