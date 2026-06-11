import cv2
import os

# --- CẤU HÌNH THƯ MỤC ---
IMG_DIR = "face_detection_dataset/images"
LBL_DIR = "face_detection_dataset/labels"
LOG_DIR = "face_detection_dataset/logs"

os.makedirs(LBL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- LOAD CÁC MODEL NHẬN DIỆN ---
# 1. Model mat chinh dien (Mac dinh)
frontal_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# 2. Model mat nghieng (Profile)
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

# Kiem tra model co load duoc khong
if frontal_cascade.empty() or profile_cascade.empty():
    print("LOI: Khong tim thay file XML cua OpenCV. Kiem tra lai cai dat thu vien.")
    exit()

def label_path_for_image(img_name: str) -> str:
    base = os.path.splitext(img_name)[0]
    return os.path.join(LBL_DIR, base + ".txt")

no_face_list = []
count_frontal = 0
count_profile = 0

print("Dang chay Auto Labeling (Ho tro goc nghieng)...")
for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(IMG_DIR, img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- CHIEN THUAT 3 BUOC ---
    # Buoc 1: Quet chinh dien
    faces = frontal_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )
    detected_type = "frontal"
    
    # Buoc 2: Neu khong thay, quet nghieng (Profile)
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
        )
        detected_type = "profile"
        
    # Buoc 3: Neu van khong thay, Lat nguoc anh (Flip) va quet nghieng lan nua
    # (Vi Profile Cascade thuong chi bat duoc 1 ben nghieng)
    if len(faces) == 0:
        gray_flipped = cv2.flip(gray, 1) # 1 la lat theo truc doc
        faces_flipped = profile_cascade.detectMultiScale(
            gray_flipped, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces_flipped) > 0:
            detected_type = "profile_flipped"
            h_img, w_img = gray.shape
            # Tinh lai toa do tu anh da lat ve anh goc
            # x_moi = chieu_rong_anh - x_lat - w
            faces = []
            for (x, y, w, h) in faces_flipped:
                faces.append((w_img - x - w, y, w, h))
                
    lbl_path = label_path_for_image(img_name)
    
    # --- GHI KET QUA ---
    if len(faces) == 0:
        no_face_list.append(img_name)
        open(lbl_path, "w", encoding="utf-8").close() # Tao file rong
    else:
        if detected_type == "frontal":
            count_frontal += 1
        else:
            count_profile += 1
        with open(lbl_path, "w", encoding="utf-8") as f:
            for (x, y, w, h) in faces:
                f.write(f"{x} {y} {w} {h}\n")

# --- GHI LOG VA BAO CAO ---
log_path = os.path.join(LOG_DIR, "no_face_images.txt")
with open(log_path, "w", encoding="utf-8") as f:
    for name in no_face_list:
        f.write(name + "\n")

print("-" * 30)
print("HOAN TAT!")
print(f"Tong so anh: {len(os.listdir(IMG_DIR))}")
print(f"- Bat duoc chinh dien: {count_frontal}")
print(f"- Bat duoc goc nghieng: {count_profile}")
print(f"- Khong thay mat (Loi): {len(no_face_list)}")
print(f"File log loi: {log_path}")
print("-" * 30)
