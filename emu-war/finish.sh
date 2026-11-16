#!/bin/bash
# finish.sh — Final encoding, loudness normalization, QA
# Usage: ./finish.sh "Title"

set -e

TITLE="${1:-Emu War}"
PROJECT_DIR="/sessions/friendly-festive-bardeen/shorts/emu-war"
FRAMES_DIR="$PROJECT_DIR/frames"
OUT_DIR="$PROJECT_DIR/out"
MIX_WAV="$PROJECT_DIR/mix.wav"
FINAL_MP4="$OUT_DIR/${TITLE// /_}.mp4"
QA_SHEET="$OUT_DIR/qa_sheet.jpg"

mkdir -p "$OUT_DIR"

echo "=== Final Encoding ==="

# 1. Linear loudness gain to ~-14 LUFS + -1 dBFS peak limiter
ffmpeg -y -i "$MIX_WAV" \
  -af "loudnorm=I=-14:TP=-1:LRA=11:print_format=summary" \
  -ar 44100 "$OUT_DIR/loudnorm.wav" 2>&1 | tail -20

# 2. Encode video: H.264 CRF 17 + AAC
echo "Encoding video..."
ffmpeg -y -framerate 30 -i "$FRAMES_DIR/f%04d.jpg" \
  -i "$OUT_DIR/loudnorm.wav" \
  -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p \
  -c:a aac -b:a 192k -ar 44100 \
  -movflags +faststart \
  "$FINAL_MP4"

echo "Video encoded: $FINAL_MP4"

# 3. QA: Freeze/black frame detection
echo "Running QA..."
python3 << 'PYEOF'
import subprocess
import cv2
import numpy as np
import os

FRAMES_DIR = "/sessions/friendly-festive-bardeen/shorts/emu-war/frames"
OUT_DIR = "/sessions/friendly-festive-bardeen/shorts/emu-war/out"
QA_SHEET = os.path.join(OUT_DIR, "qa_sheet.jpg")

files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
print(f"Checking {len(files)} frames...")

frozen = []
black = []
prev = None
freeze_count = 0

for i, f in enumerate(files):
    img = cv2.imread(os.path.join(FRAMES_DIR, f))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)

    if mean_val < 5:
        black.append(i)

    if prev is not None:
        diff = np.mean(np.abs(gray.astype(float) - prev.astype(float)))
        if diff < 1.0:
            freeze_count += 1
        else:
            if freeze_count >= 12:
                frozen.append((i - freeze_count, freeze_count))
            freeze_count = 0
    prev = gray

if freeze_count >= 12:
    frozen.append((len(files) - freeze_count, freeze_count))

print(f"Frozen spans (>=12 frames): {len(frozen)}")
for start, dur in frozen:
    print(f"  Frames {start}-{start+dur-1} ({dur/30:.2f}s)")

print(f"Black frames: {len(black)}")
for b in black[:10]:
    print(f"  Frame {b}")
if len(black) > 10:
    print(f"  ... and {len(black)-10} more")

# Create QA sheet
cols = 10
sample_indices = list(range(0, len(files), max(1, len(files) // (cols * 6))))
sample_indices = sample_indices[:cols * 6]

import PIL.Image
first = PIL.Image.open(os.path.join(FRAMES_DIR, files[0]))
w, h = first.size
sheet = PIL.Image.new('RGB', (w * cols, h * 6), (20, 20, 20))

for idx, fi in enumerate(sample_indices):
    img = PIL.Image.open(os.path.join(FRAMES_DIR, files[fi]))
    if fi in black or any(s <= fi < s+d for s,d in frozen):
        overlay = PIL.Image.new('RGBA', img.size, (255, 0, 0, 80))
        img = img.convert('RGBA')
        img = PIL.Image.alpha_composite(img, overlay).convert('RGB')
    x = (idx % cols) * w
    y = (idx // cols) * h
    sheet.paste(img, (x, y))

sheet.save(QA_SHEET, quality=85)
print(f"QA sheet saved: {QA_SHEET}")

if frozen or black:
    print("ISSUES FOUND - Review QA sheet")
    exit(1)
else:
    print("QA PASSED")
    exit(0)
PYEOF

QA_RESULT=$?
if [ $QA_RESULT -ne 0 ]; then
    echo "QA failed. Check $QA_SHEET"
    exit 1
fi

echo "=== DONE ==="
echo "Final video: $FINAL_MP4"
echo "QA sheet: $QA_SHEET"
