import cv2
import os

IMG_DIR = "face_detection_dataset/images"
LBL_DIR = "face_detection_dataset/labels"
LOG_DIR = "face_detection_dataset/logs"

os.makedirs(LBL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def label_path_for_image(img_name: str) -> str:
    base = os.path.splitext(img_name)[0]
    return os.path.join(LBL_DIR, base + ".txt")

no_face_list = []

for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(IMG_DIR, img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    lbl_path = label_path_for_image(img_name)
    if len(faces) == 0:
        no_face_list.append(img_name)
        # create empty label file for synchronization
        open(lbl_path, "w", encoding="utf-8").close()
        continue
        
    with open(lbl_path, "w", encoding="utf-8") as f:
        for (x, y, w, h) in faces:
            f.write(f"{x} {y} {w} {h}\n")

# write log
log_path = os.path.join(LOG_DIR, "no_face_images.txt")
with open(log_path, "w", encoding="utf-8") as f:
    for name in no_face_list:
        f.write(name + "\n")

print("Created labels for images in:", LBL_DIR)
print("Number of images with no face detected:", len(no_face_list))
print("List saved at:", log_path)
