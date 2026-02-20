"""
DeepShield — Dataset Preparation Script
========================================
Downloads FaceForensics++ (Kaggle mirror) and optionally DFDC video frames,
then organises them into the standard structure:

    data/
        real/   ← authentic face images
        fake/   ← deepfake / AI-generated images

Usage:
    # Basic (FaceForensics++ only, 5000 images per class):
    python scripts/prepare_dataset.py --output_dir ./data

    # With DFDC frames as well:
    python scripts/prepare_dataset.py --output_dir ./data --include_dfdc

    # Limit size for quick tests:
    python scripts/prepare_dataset.py --output_dir ./data --max_per_class 500

Prerequisites:
    pip install kagglehub mediapipe
    Set up Kaggle credentials:
        mkdir -p ~/.config/kaggle
        cp ~/Downloads/kaggle.json ~/.config/kaggle/kaggle.json
        chmod 600 ~/.config/kaggle/kaggle.json
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

# ── helpers ──────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _collect_images(src: Path) -> list[Path]:
    """Recursively collect all supported image paths under *src*."""
    return [
        p for p in src.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]


def _copy_with_progress(files: list[Path], dst: Path, label: str, limit: int | None = None):
    """Copy *files* into *dst*, optionally capped to *limit* items."""
    dst.mkdir(parents=True, exist_ok=True)
    if limit and len(files) > limit:
        files = random.sample(files, limit)
    for src_path in tqdm(files, desc=f"  Copying {label}", unit="img"):
        dest_file = dst / src_path.name
        # Avoid name collisions by prefixing with parent dir name
        if dest_file.exists():
            dest_file = dst / f"{src_path.parent.name}_{src_path.name}"
        shutil.copy2(src_path, dest_file)
    return len(files)


def _face_crop_images(img_dir: Path, workers: int = 4):
    """
    Optional: crop faces from all images using MediaPipe.
    Improves training accuracy by focusing on the face region.
    Skips images where no face is detected.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("  ⚠ mediapipe not installed — skipping face crop step.")
        return

    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    images = list(img_dir.glob("*"))
    removed = 0
    for img_path in tqdm(images, desc="  Face-cropping", unit="img"):
        if img_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                img_path.unlink()
                removed += 1
                continue
            h, w = img.shape[:2]
            results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.detections:
                img_path.unlink()
                removed += 1
                continue
            det = results.detections[0].location_data.relative_bounding_box
            x1 = max(int(det.xmin * w) - 20, 0)
            y1 = max(int(det.ymin * h) - 20, 0)
            x2 = min(int((det.xmin + det.width) * w) + 20, w)
            y2 = min(int((det.ymin + det.height) * h) + 20, h)
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                img_path.unlink()
                removed += 1
                continue
            cv2.imwrite(str(img_path), face)
        except Exception:
            img_path.unlink(missing_ok=True)
            removed += 1

    face_detector.close()
    print(f"  Face crop done — removed {removed} images without detectable faces.")


# ── dataset downloaders ───────────────────────────────────────────────────────

def download_ff_plus_plus(output_dir: Path, max_per_class: int | None):
    """
    Download 'manjilkarki/deepfake-and-real-images' from Kaggle.
    This dataset contains FaceForensics++ derived images (~190k total).
    Dataset structure:
        Dataset/
            Real/   ← real face photos
            Fake/   ← deepfake images (FaceSwap, Deepfakes, etc.)
    """
    print("\n[1/2] Downloading FaceForensics++ Kaggle mirror...")
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    try:
        path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        print("Make sure your Kaggle credentials are set up:")
        print("  mkdir -p ~/.config/kaggle")
        print("  cp ~/Downloads/kaggle.json ~/.config/kaggle/kaggle.json")
        print("  chmod 600 ~/.config/kaggle/kaggle.json")
        sys.exit(1)

    src = Path(path)
    print(f"  Downloaded to: {src}")

    # Locate real and fake subdirs (handle nested structure)
    real_srcs = list(src.rglob("Real")) + list(src.rglob("real"))
    fake_srcs = list(src.rglob("Fake")) + list(src.rglob("fake"))

    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"

    all_real = []
    for d in real_srcs:
        if d.is_dir():
            all_real.extend(_collect_images(d))

    all_fake = []
    for d in fake_srcs:
        if d.is_dir():
            all_fake.extend(_collect_images(d))

    print(f"  Found {len(all_real):,} real images, {len(all_fake):,} fake images")

    n_real = _copy_with_progress(all_real, real_dir, "FF++ real", max_per_class)
    n_fake = _copy_with_progress(all_fake, fake_dir, "FF++ fake", max_per_class)

    print(f"  ✅ FaceForensics++: {n_real} real, {n_fake} fake copied.")
    return n_real, n_fake


def download_dfdc(output_dir: Path, max_per_class: int | None):
    """
    Download DFDC (DeepFake Detection Challenge) video frame images from Kaggle.
    Uses dataset: 'dagnelies/deepfake-faces' which contains pre-extracted frames.
    """
    print("\n[2/2] Downloading DFDC face frames...")
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed.")
        sys.exit(1)

    try:
        path = kagglehub.dataset_download("dagnelies/deepfake-faces")
    except Exception as e:
        print(f"  ⚠ DFDC download failed: {e}")
        print("  Skipping DFDC — continuing with FaceForensics++ only.")
        return 0, 0

    src = Path(path)
    print(f"  Downloaded to: {src}")

    real_srcs = list(src.rglob("real")) + list(src.rglob("REAL"))
    fake_srcs = list(src.rglob("fake")) + list(src.rglob("FAKE")) + list(src.rglob("manipulated"))

    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"

    all_real = []
    for d in real_srcs:
        if d.is_dir():
            all_real.extend(_collect_images(d))

    all_fake = []
    for d in fake_srcs:
        if d.is_dir():
            all_fake.extend(_collect_images(d))

    print(f"  Found {len(all_real):,} real frames, {len(all_fake):,} fake frames")

    n_real = _copy_with_progress(all_real, real_dir, "DFDC real", max_per_class // 2 if max_per_class else None)
    n_fake = _copy_with_progress(all_fake, fake_dir, "DFDC fake", max_per_class // 2 if max_per_class else None)

    print(f"  ✅ DFDC: {n_real} real, {n_fake} fake copied.")
    return n_real, n_fake


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare deepfake training datasets"
    )
    parser.add_argument(
        "--output_dir", default="./data",
        help="Where to save the prepared dataset (default: ./data)"
    )
    parser.add_argument(
        "--max_per_class", type=int, default=None,
        help="Max images per class. Omit for full dataset."
    )
    parser.add_argument(
        "--include_dfdc", action="store_true",
        help="Also download DFDC face frames"
    )
    parser.add_argument(
        "--face_crop", action="store_true",
        help="Crop faces using MediaPipe after download (recommended for better accuracy)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DeepShield — Dataset Preparation")
    print("=" * 60)
    print(f"  Output: {output_dir.resolve()}")
    if args.max_per_class:
        print(f"  Limit:  {args.max_per_class} images per class")
    print()

    total_real, total_fake = 0, 0

    # Download FaceForensics++ (primary dataset)
    r, f = download_ff_plus_plus(output_dir, args.max_per_class)
    total_real += r
    total_fake += f

    # Optionally add DFDC
    if args.include_dfdc:
        r, f = download_dfdc(output_dir, args.max_per_class)
        total_real += r
        total_fake += f

    # Optional face crop step
    if args.face_crop:
        print("\n[Face Crop] Cropping faces from real images...")
        _face_crop_images(output_dir / "real")
        print("[Face Crop] Cropping faces from fake images...")
        _face_crop_images(output_dir / "fake")

    # Final summary
    final_real = len(_collect_images(output_dir / "real"))
    final_fake = len(_collect_images(output_dir / "fake"))

    print()
    print("=" * 60)
    print("  Dataset Ready!")
    print("=" * 60)
    print(f"  Real images : {final_real:,}")
    print(f"  Fake images : {final_fake:,}")
    print(f"  Total       : {final_real + final_fake:,}")
    print()
    print("  Next step — run training:")
    print(f"  python train/train_all.py --data_dir {output_dir} --epochs 20")
    print()

    # Warn about imbalance
    if abs(final_real - final_fake) / max(final_real + final_fake, 1) > 0.2:
        print("  ⚠ WARNING: Class imbalance detected (>20% difference).")
        print("    Consider using --max_per_class to balance the dataset.")


if __name__ == "__main__":
    main()
