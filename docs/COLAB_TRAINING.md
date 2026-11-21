# Training on Google Colab

Takes about 30 minutes in total, about 20 of them training.

## What you need
| File | Where |
|---|---|
| `dist/speed_bump_yolo.zip` (79 MB): images + labels + config | in this project |
| `notebooks/train_colab.ipynb`: the training notebook | in this project |

## Step by step

1. **Upload the data to Google Drive.**
   In [drive.google.com](https://drive.google.com), create a folder **`speed-bump`** in *My Drive* and upload `speed_bump_yolo.zip` into it, so you get `MyDrive/speed-bump/speed_bump_yolo.zip`.
   Don't unzip it. The notebook does that.
2. **Open the notebook.** Go to [colab.research.google.com](https://colab.research.google.com), then *File → Upload notebook*, and choose `notebooks/train_colab.ipynb`.
3. **Turn on the GPU.** *Runtime → Change runtime type → T4 GPU → Save.*
4. **Run the cells.** Use *Runtime → Run all*, or run the cells one by one. Approve the Google Drive access pop-up in step 3 of the notebook.
5. **Read the results** (steps 6–8 of the notebook):
   - per-class precision, recall and mAP50 on the **test** split
   - the % of bump-free road images that trigger a false detection
   - sample predictions, training curves and the confusion matrix
6. **Collect the outputs.** Everything is saved to `MyDrive/speed-bump/runs/yolo11n_v1/`. `weights/best.pt` is the trained model and `weights/best.onnx` is the export.

**If Colab disconnects:** the checkpoints are on Drive. Rerun steps 1–3, then run the **resume** cell under step 5.

## What's in the data (and how labels get in)

The zip is built from your labels by a short pipeline:

```
data/clean/labels_manual/*.txt   ← your hand-made / CVAT labels (highest priority)
data/clean/labels_draft/*.txt    ← automatic drafts (used only if reviewed "ok")
data/clean/label_overrides.csv   ← review decisions for the drafts
        │  python src/data/merge_labels.py
        ▼
data/clean/labels + label_review.csv   (status of every image)
        │  python src/data/build_yolo.py
        ▼
data/yolo/   (train/val/test, images + labels + bumps.yaml)
        │  python src/data/package_for_colab.py
        ▼
dist/speed_bump_yolo.zip   → upload to Drive
```

Label format (YOLO): one `.txt` per image, one line per bump:
`class x_center y_center width height`, all relative to the image size (0–1). Class `0` = marked bump, `1` = unmarked bump. An **empty file** means "no bump in this image".

**Current contents:** 268 marked and 29 unmarked boxed images, plus 597 road images with no bump.
170 images are left out because they still need a box (see `data/clean/label_review.csv`).

### Adding more labels (do this before the next training run)

1. Install [CVAT](https://app.cvat.ai) (free online account) or Label Studio.
2. Create a task with two labels, `marked_bump` and `unmarked_bump`, in that order.
3. Upload the images from `data/clean/images/` whose status in `label_review.csv` is `needs_label` or `needs_fix`.
   For `needs_fix` images you can import their draft box from `data/clean/labels/` (*Upload annotations → YOLO 1.1*), then correct it.
4. Draw boxes following [labeling_guidelines.md](labeling_guidelines.md). If an image has no visible bump, leave it empty and add the line `<file>,drop` to `data/clean/label_overrides.csv`.
5. *Export → YOLO 1.1*, then copy the exported `.txt` files into **`data/clean/labels_manual/`**. Names must match the images, e.g. `u_image40.txt`.
6. Rebuild and re-upload:
   ```bash
   python src/data/merge_labels.py
   python src/data/build_yolo.py
   python src/data/package_for_colab.py
   ```
   Then replace the zip on Drive, change `name='yolo11n_v1'` to `'yolo11n_v2'` in the notebook, and run it again.

**Adding brand-new photos or videos:** put them in a new folder under `bumps-dataset/`, add that folder to `SOURCES` in `src/data/clean.py`, and add their scene groups to `src/data/scene_groups.py`. All shots of the same bump must share a group. Then run the whole pipeline (`clean → split → merge_labels → build_yolo → package`).

## Reading the results

| Number (notebook step) | Good for a first model | If it's bad |
|---|---|---|
| Marked recall (6) | ≥ 0.85 | More marked data at a distance; try `imgsz=960` |
| Unmarked recall (6) | Anything above 0 is a start. The test split has only 4 unmarked images | **Label more unmarked bumps.** This is the main bottleneck |
| False alarms on background (7) | ≤ 5 % | Add hard negatives: zebra crossings, shadows, patched asphalt |
| mAP50 overall (6) | ≥ 0.7 | Check the confusion matrix to see which class fails |

Keep a short log of each run so you can compare versions:

| run | data version | model | marked R | unmarked R | false-alarm % | notes |
|---|---|---|---|---|---|---|
| yolo11n_v1 | 297 boxed + 597 bg | yolo11n, 640 | | | | baseline |

## What to do next (in order)

1. **Run the baseline and record its numbers.** It's the reference point for everything after it.
2. **Label the 64 unmarked + 106 marked images still pending.** Unmarked data gives the biggest gain. Aim for ≥ 150 unmarked boxes.
3. **Record a dashcam test drive** (phone on the windscreen, ~1.2 m high) over known bumps and bump-free roads. Run notebook step 9 on it. This is the real-world test that matters.
4. **Add hard negatives** from that drive: frames with zebra crossings and shadows, saved with empty labels.
5. **Try `yolo11s.pt`** and `imgsz=960`, and compare against the baseline on the same test split.
6. **Build the real-time demo**: webcam or video plus the ONNX model, with an alert only when a bump is seen in ≥ 3 of 5 consecutive frames (guide §5, Phase 4).
7. **Deploy:** uncomment the TFLite export in notebook step 10 for Android, or use ONNX on a laptop.
