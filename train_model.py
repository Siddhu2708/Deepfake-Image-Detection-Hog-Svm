import os
import cv2
import numpy as np
import joblib

from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# =========================================================
# SETTINGS
# =========================================================

DATASET_DIR = "dataset"

REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "AI")

MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_svm.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

IMAGE_SIZE = (128, 128)

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# HOG FEATURE EXTRACTION
# =========================================================

def extract_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    # Resize
    image = cv2.resize(image, IMAGE_SIZE)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# =========================================================
# LOAD DATASET
# =========================================================

X = []
y = []

print("\n======================================")
print(" DEEPFAKE IMAGE DETECTION")
print("======================================")

print("\nLoading REAL images...")

real_count = 0

for filename in os.listdir(REAL_DIR):

    file_path = os.path.join(REAL_DIR, filename)

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    ):

        features = extract_features(file_path)

        if features is not None:

            X.append(features)

            # REAL = 0
            y.append(0)

            real_count += 1


print("REAL images loaded:", real_count)


print("\nLoading AI / FAKE images...")

fake_count = 0

for filename in os.listdir(FAKE_DIR):

    file_path = os.path.join(FAKE_DIR, filename)

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    ):

        features = extract_features(file_path)

        if features is not None:

            X.append(features)

            # FAKE = 1
            y.append(1)

            fake_count += 1


print("AI / FAKE images loaded:", fake_count)


# =========================================================
# CHECK DATA
# =========================================================

if len(X) == 0:

    print("\nERROR: No images found!")

    print("Check this structure:")
    print("""
    dataset/
        real/
        AI/
    """)

    exit()


X = np.array(X)
y = np.array(y)


print("\n======================================")
print(" DATASET INFORMATION")
print("======================================")

print("Total images:", len(X))
print("Real images :", real_count)
print("Fake images :", fake_count)
print("Feature size:", X.shape[1])


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining images:", len(X_train))
print("Testing images :", len(X_test))


# =========================================================
# FEATURE SCALING
# =========================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# =========================================================
# SVM MODEL
# =========================================================

print("\n======================================")
print(" TRAINING SVM MODEL")
print("======================================")

model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)


# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print(" MODEL RESULT")
print("======================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["REAL", "FAKE"]
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(model, MODEL_PATH)

joblib.dump(scaler, SCALER_PATH)


print("\n======================================")
print(" MODEL SAVED")
print("======================================")

print("Model :", MODEL_PATH)
print("Scaler:", SCALER_PATH)

print("\nTraining completed successfully!")