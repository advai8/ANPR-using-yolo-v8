import cv2
from ultralytics import YOLO
import easyocr
from datetime import datetime, timezone
import re
import time
from supabase import create_client
from collections import deque, Counter
import uuid

print("Running CPU stable mode with proof upload")

# -------------------- SUPABASE --------------------
SUPABASE_URL = "YOUR_URL_HERE"
SUPABASE_KEY = "YOUR_KEY_HERE"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- CONFIG --------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 384
FRAME_HEIGHT = 288
CONFIDENCE_THRESHOLD = 0.7
EXIT_DELAY_SECONDS = 30
PLATE_COOLDOWN = 1
FRAME_VOTE_COUNT = 2

# -------------------- MODEL --------------------
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Camera not found")
    exit()

recently_processed = {}
plate_buffer = deque(maxlen=FRAME_VOTE_COUNT)
plate_state_cache = {}
frame_counter = 0

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
        else:
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
        print("Uploaded with proof to Supabase")

    except Exception as e:
        print("Supabase Error:", e)

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    results = model(frame, imgsz=384, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy
        confidences = r.boxes.conf

        for box, conf in zip(boxes, confidences):
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)

            frame_counter += 1
            if frame_counter % 3 != 0:
                continue

            plate_img = frame[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
            ocr_results = reader.readtext(plate_img, detail=0)

            if not ocr_results:
                continue

            plate_number = clean_plate(''.join(ocr_results))

            if not valid_plate(plate_number):
                continue

            plate_buffer.append(plate_number)

            if len(plate_buffer) < FRAME_VOTE_COUNT:
                continue

            plate_number = Counter(plate_buffer).most_common(1)[0][0]
            plate_buffer.clear()

            current_time = time.time()

            if plate_number in recently_processed:
                if current_time - recently_processed[plate_number] < PLATE_COOLDOWN:
                    continue

            recently_processed[plate_number] = current_time

            last_status, last_time = get_last_record(plate_number)
            now_utc = datetime.now(timezone.utc)

            if last_status is None:
                status = "ENTRY"
            elif last_status == "ENTRY":
                if (now_utc - last_time).total_seconds() >= EXIT_DELAY_SECONDS:
                    status = "EXIT"
                else:
                    continue
            else:
                status = "ENTRY"

            # Upload proof
            plate_path = upload_image(plate_img, "plates")
            frame_path = upload_image(frame, "frames")

            print(f"{plate_number} | {status}")

            send_to_supabase(plate_number, status, plate_path, frame_path)

    cv2.imshow("ANPR Bus Entry/Exit Logging System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()