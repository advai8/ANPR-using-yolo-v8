import cv2
from ultralytics import YOLO
import easyocr
from datetime import datetime, timezone
import re
import time
from supabase import create_client
from collections import deque, Counter
import uuid
import threading
import concurrent.futures

print("Running CPU stable mode with proof upload")

# -------------------- SUPABASE --------------------
SUPABASE_URL = "YOUR URL HERE"
SUPABASE_KEY = "YOUR KEY HERE"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- CONFIG --------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640  # Increased for better OCR resolution, while YOLO acts fast at 384
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.6 # Optimized for stable capture
EXIT_DELAY_SECONDS = 30
PLATE_COOLDOWN = 3  # Cooldown between processing the same plate
FRAME_VOTE_COUNT = 2

# -------------------- VIDEO STREAM THREAD --------------------
class VideoStream:
    """Threaded video capture to eliminate OpenCV buffer lag."""
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame
            # Short yield reduces CPU core lock
            time.sleep(0.005)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# -------------------- BACKGROUND WORKERS --------------------
# ThreadPoolExecutor to prevent network IO (Supabase) from freezing the camera feed
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

# -------------------- MODEL INIT --------------------
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'])
print("Models loaded successfully.")

stream = VideoStream(CAMERA_INDEX)
if not stream.ret:
    print("Camera not found")
    exit()

recently_processed = {}
plate_buffer = deque(maxlen=FRAME_VOTE_COUNT)
plate_state_cache = {}

# -------------------- HELPERS --------------------
def clean_plate(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    text = text.replace('O', '0')
    text = text.replace('I', '1')
    text = text.replace('Z', '2')
    return text

def valid_plate(text):
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,4}$'
    return re.match(pattern, text)

def upload_image(image, folder):
    filename = f"{folder}/{uuid.uuid4()}.jpg"
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        return None

    try:
        supabase.storage.from_("plate-evidence").upload(
            filename,
            buffer.tobytes(),
            {"content-type": "image/jpeg"}
        )
        return filename
    except Exception as e:
        print("Storage Upload Error:", e)
        return None

def get_last_record(plate_number):
    if plate_number in plate_state_cache:
        return plate_state_cache[plate_number]
    try:
        response = supabase.table("logs") \
            .select("status, timestamp") \
            .eq("bus_number", plate_number) \
            .order("timestamp", desc=True) \
            .limit(1) \
            .execute()
        if response.data:
            last_status = response.data[0]["status"]
            last_timestamp = response.data[0]["timestamp"]
            last_time = datetime.fromisoformat(last_timestamp.replace("Z", "+00:00"))
            plate_state_cache[plate_number] = (last_status, last_time)
            return last_status, last_time
        return None, None
    except Exception as e:
        print("Supabase Error:", e)
        return None, None

def send_to_supabase(plate_number, status, plate_path, frame_path):
    now_utc = datetime.now(timezone.utc)
    try:
        supabase.table("logs").insert({
            "bus_number": plate_number,
            "status": status,
            "timestamp": now_utc.isoformat(),
            "plate_image": plate_path,
            "frame_image": frame_path
        }).execute()
        plate_state_cache[plate_number] = (status, now_utc)
        print(f"Successfully processed event: {plate_number} -> {status}")
    except Exception as e:
        print("Supabase Error:", e)

def background_upload_task(plate_number, plate_img, frame_img, now_utc):
    """Executes network calls in a background thread."""
    try:
        last_status, last_time = get_last_record(plate_number)
        
        if last_status is None:
            status = "ENTRY"
        elif last_status == "ENTRY":
            if (now_utc - last_time).total_seconds() >= EXIT_DELAY_SECONDS:
                status = "EXIT"
            else:
                return # Cooldown for exit logic hasn't elapsed
        else:
            status = "ENTRY"

        print(f"Uploading snapshot for {plate_number} ({status})...")
        plate_path = upload_image(plate_img, "plates")
        frame_path = upload_image(frame_img, "frames")

        if plate_path and frame_path:
            send_to_supabase(plate_number, status, plate_path, frame_path)
    except Exception as e:
        print(f"Background upload error: {e}")

# -------------------- MAIN LOOP --------------------
print("Ready. Press 'q' to quit.")
prev_time = time.time()

while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        continue

    # Create a display copy so OCR extraction works on raw frame without UI overlays
    display_frame = frame.copy()

    # YOLO Inference - resize to imgsz=384 internally for speed, 
    # but bounding boxes will map to original 640x480 frame for crisp OCR
    results = model(frame, imgsz=384, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy
        confidences = r.boxes.conf

        for box, conf in zip(boxes, confidences):
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)
            
            # Draw tracking bounding box on UI
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Conf: {conf:.2f}", (x1, max(y1 - 25, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Extract license plate
            plate_img = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            
            # Skip invalid extractions
            if plate_img.shape[0] < 10 or plate_img.shape[1] < 20:
                continue

            # Run OCR
            ocr_results = reader.readtext(plate_img, detail=0)
            if not ocr_results:
                continue

            plate_number = clean_plate(''.join(ocr_results))
            if not valid_plate(plate_number):
                continue

            # Overlay detected plate number
            cv2.putText(display_frame, plate_number, (x1, max(y1 - 5, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            # Consensus mechanism
            plate_buffer.append(plate_number)
            if len(plate_buffer) < FRAME_VOTE_COUNT:
                continue

            plate_number_voted = Counter(plate_buffer).most_common(1)[0][0]
            plate_buffer.clear()

            current_time = time.time()
            if plate_number_voted in recently_processed:
                if current_time - recently_processed[plate_number_voted] < PLATE_COOLDOWN:
                    continue

            # Trigger network interaction asynchronously 
            recently_processed[plate_number_voted] = current_time
            now_utc = datetime.now(timezone.utc)
            
            executor.submit(background_upload_task, plate_number_voted, plate_img.copy(), frame.copy(), now_utc)

    # Frame Rate Metrics
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Advanced ANPR Pipeline", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
executor.shutdown(wait=False)
cv2.destroyAllWindows()