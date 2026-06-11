import cv2
import os

# --- DIRECTORY CONFIGURATION ---
IMG_DIR = "face_detection_dataset/images"
LBL_DIR = "face_detection_dataset/labels"
LOG_DIR = "face_detection_dataset/logs"

os.makedirs(LBL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- LOAD DETECTION MODELS ---
# 1. Frontal face model (Default)
frontal_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# 2. Profile face model
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

# Check if models are loaded successfully
if frontal_cascade.empty() or profile_cascade.empty():
    print("ERROR: OpenCV XML files not found. Please check library installation.")
    exit()

def label_path_for_image(img_name: str) -> str:
    base = os.path.splitext(img_name)[0]
    return os.path.join(LBL_DIR, base + ".txt")

no_face_list = []
count_frontal = 0
count_profile = 0

print("Running Auto Labeling (Supporting profile faces)...")
for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(IMG_DIR, img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- 3-STEP STRATEGY ---
    # Step 1: Scan frontal face
    faces = frontal_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )
    detected_type = "frontal"
    
    # Step 2: If not found, scan profile face
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
        )
        detected_type = "profile"
        
    # Step 3: If still not found, flip image and scan profile again
    # (Because Profile Cascade usually only detects one side)
    if len(faces) == 0:
        gray_flipped = cv2.flip(gray, 1) # 1 is to flip horizontally
        faces_flipped = profile_cascade.detectMultiScale(
            gray_flipped, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces_flipped) > 0:
            detected_type = "profile_flipped"
            h_img, w_img = gray.shape
            # Recalculate coordinates from flipped image to original image
            # new_x = image_width - flipped_x - w
            faces = []
            for (x, y, w, h) in faces_flipped:
                faces.append((w_img - x - w, y, w, h))
                
    lbl_path = label_path_for_image(img_name)
    
    # --- SAVE RESULTS ---
    if len(faces) == 0:
        no_face_list.append(img_name)
        open(lbl_path, "w", encoding="utf-8").close() # Create empty file
    else:
        if detected_type == "frontal":
            count_frontal += 1
        else:
            count_profile += 1
        with open(lbl_path, "w", encoding="utf-8") as f:
            for (x, y, w, h) in faces:
                f.write(f"{x} {y} {w} {h}\n")

# --- SAVE LOGS AND REPORTS ---
log_path = os.path.join(LOG_DIR, "no_face_images.txt")
with open(log_path, "w", encoding="utf-8") as f:
    for name in no_face_list:
        f.write(name + "\n")

print("-" * 30)
print("COMPLETED!")
print(f"Total images: {len(os.listdir(IMG_DIR))}")
print(f"- Detected frontal faces: {count_frontal}")
print(f"- Detected profile faces: {count_profile}")
print(f"- No face detected (Errors): {len(no_face_list)}")
print(f"Error log file: {log_path}")
print("-" * 30)
