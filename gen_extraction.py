import zipfile
import kaggle
import os

# -----------------------------
# CONFIG
# -----------------------------
KAGGLE_DATASET = "shinigdapriyasathish/images-and-its-captions-of-indian-streets"    # e.g., "ultralytics/yolov5"
ZIP_NAME = "images-and-its-captions-of-indian-streets.zip"                       # kaggle will save here

SAVE_DIR = r"D:\ANNDL\dd"             # where extracted files go
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_IMAGES = 2000   # how many images you want
ALLOWED_EXT = [".jpg", ".jpeg", ".png"]     # allowed image types

# OPTIONAL filtering by folder names
ALLOWED_FOLDERS = ["train", "images", "idd"]   # customize for each dataset


# -----------------------------
# 1. DOWNLOAD ZIP FILE FROM KAGGLE
# -----------------------------
print("\n📥 Downloading dataset ZIP from Kaggle ...")
kaggle.api.dataset_download_files(
    KAGGLE_DATASET,
    path=".",
    force=True,
    quiet=False
)

print("\n✔ Download complete")


# -----------------------------
# 2. SELECTIVE EXTRACTION
# -----------------------------
print("\n📤 Selectively extracting files ...")

count = 0

with zipfile.ZipFile(ZIP_NAME, "r") as z:
    for file in z.namelist():

        # only images
        if not file.lower().endswith(tuple(ALLOWED_EXT)):
            continue

        # filter by folder name (optional)
        if not any(folder in file.lower() for folder in ALLOWED_FOLDERS):
            continue

        # stop after MAX_IMAGES
        if count >= MAX_IMAGES:
            break

        # extract to custom directory
        z.extract(file, SAVE_DIR)
        count += 1

        if count % 500 == 0:
            print(f"Extracted: {count}")

print(f"\n🎉 DONE! Extracted {count} images into {SAVE_DIR}")
