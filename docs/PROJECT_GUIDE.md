# Real-Time Speed Bump Detector: Project Guide

> **Goal:** a computer-vision model that spots speed bumps (marked and unmarked) in a live road camera feed early enough to warn the driver.
>
> **Status (2026-11-19, updated):** dataset cleaned, split by scene, 297/467 images boxed, 597 negatives added, YOLO dataset built.
> See [README.md](../README.md) for exactly what was fixed and what is left. The audit below describes the **raw** data.
> This guide covers what is in the dataset, what to fix first, which stack to use, a phased plan, and features to add later.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Dataset audit (what you actually have)](#2-dataset-audit-what-you-actually-have)
3. [How to frame the problem](#3-how-to-frame-the-problem)
4. [Recommended stack](#4-recommended-stack)
5. [Development roadmap](#5-development-roadmap)
6. [Evaluation: what "good" means](#6-evaluation-what-good-means)
7. [Deployment targets](#7-deployment-targets)
8. [Features to add later](#8-features-to-add-later)
9. [What similar projects did](#9-what-similar-projects-did)
10. [Risks and pitfalls](#10-risks-and-pitfalls)
11. [Suggested repository layout](#11-suggested-repository-layout)
12. [Next steps checklist](#12-next-steps-checklist)
13. [References](#13-references)

---

## 1. TL;DR

| Question | Answer |
|---|---|
| What kind of model? | **Object detection** with 2 classes: `marked_bump`, `unmarked_bump`. Not image classification. |
| Which model to start with? | **YOLO11n / YOLO11s** (Ultralytics), pretrained on COCO and fine-tuned. Move to **RF-DETR** or **D-FINE** (Apache-2.0) if you need a licence that allows commercial use. |
| Biggest blocker today | The dataset has **no bounding boxes**, **76 images are in both class folders**, and there are **no images without a bump**. Fix the data before training. |
| Minimum data to aim for | About 1,500 labelled bump images (at least 400 unmarked) plus about 1,500 images or frames with no bump, including hard negatives such as zebra crossings. |
| Where to train | Free Colab or Kaggle GPU (T4/P100). A 4 GB laptop GPU can train YOLO11n/s at `imgsz=640, batch=8`. |
| Where to run it | Android phone (TFLite/NCNN), Jetson (TensorRT) or a laptop with a webcam (ONNX Runtime). |
| Best feature to add | Fusing the phone's GPS and accelerometer with the camera. You get warnings based on distance and time-to-bump, a shared bump map, and labels collected automatically. |

---

## 2. Dataset audit (what you actually have)

I scanned every file in `bumps-dataset/` (hashes, EXIF, resolutions) and looked at random samples.

### 2.1 Raw numbers

| Folder | Files | Notes |
|---|---:|---|
| `Marked Hump/Bump/` | 450 | 374 × `Image000NN.jpg` (painted bumps) + **76 × `imageN.jpg`** |
| `Unmarked bumps/` | 93 | 78 × `imageN.jpg` + 15 × `N.jpg` |
| **Total files** | **543** | |
| **Unique images** | **467** | 76 groups of byte-identical duplicates |

### 2.2 Findings

| # | Finding | Impact | Fix |
|---|---|---|---|
| 1 | **76 images are byte-identical in *both* folders** (`imageN.jpg`). Spot checks show they are **unmarked** bumps: plain asphalt, no paint. | The labels contradict each other, and the marked class is polluted. | Delete them from `Marked Hump/Bump/`. The list is in [`dataset_audit_duplicates.csv`](dataset_audit_duplicates.csv). Check a few more before deleting. |
| 2 | **No annotations.** Only folder-level labels. | Detection and localisation are impossible until boxes are drawn. | Annotate (see §5 Phase 1). |
| 3 | **No negative images.** Every image contains a bump. | The model never learns what "no bump" looks like, which causes many false alarms on real roads. | Add images or video frames of roads without bumps. |
| 4 | **Bursts of near-duplicates.** `Image00001…00400` contains gaps, and consecutive files show the same bump from a few steps apart. A perceptual hash finds about 900 near-duplicate pairs. | A random train/test split leaks scenes and inflates the metrics. | Split **by scene or group**, not by image (see §5 Phase 1). |
| 5 | **Class imbalance after the fix:** 374 marked vs 93 unmarked (4:1). Unmarked bumps are also much harder to see, often a faint ridge or a change in texture. | Unmarked recall will be poor. | Collect more unmarked data and oversample it. Report recall per class. |
| 6 | **Narrow domain.** 537 photos come from one phone (Lenovo K8 Plus), almost all daytime, dry roads, Indian roads, taken by hand at pedestrian height. | Will not generalise to dashcams, night, rain or other countries. | Add dashcam-height footage and night or rain clips, and test on footage the model has never seen. |
| 7 | **Mixed resolutions and orientations** (460×816, 816×460, 353×627, 780×1040, up to 1600×1200). | Fine for YOLO (letterboxing). | Strip EXIF rotation and normalise orientation once during preprocessing. |
| 8 | **6 files carry GPS EXIF** (`Unmarked bumps/10–15.jpg`, iPhone 6 Plus). | Privacy leak if published. | Strip EXIF before sharing the dataset. |

### 2.3 Is this dataset public already?

The folder names and counts closely match public Roboflow Universe projects:
- *speed-bump-hump-marked* has **450 images**, the same number as your `Marked Hump/Bump`.
- *Indian Road Humps* has the classes **"Marked Hump" / "Unmarked-Hump"**.

**Check them first.** If one of them is your source and already has bounding boxes, you save days of labelling. Check its licence (most are CC BY 4.0) and inspect its labels before trusting them.

---

## 3. How to frame the problem

| Formulation | What it gives you | Verdict |
|---|---|---|
| Image classification (bump / no bump) | One label per frame | Too weak. No location or distance, and it breaks when the bump is small in the frame. Useful only as a quick baseline. |
| **Object detection** (boxes) | Where the bump is → how far away it is → when to warn | **Recommended.** Real-time, mature tooling, easy to deploy. |
| Instance segmentation (masks) | Exact bump outline across the lane | Later, only if you need lane-level precision or bump-width estimates. Costs more to label. |
| Sensor-only (accelerometer) | Detects the bump **when you hit it** | No early warning on its own, but very useful for building a map and for collecting labels automatically (§8). |

**Classes (v1):** `marked_bump`, `unmarked_bump`.
**Later candidates:** `pothole`, `rumble_strip`, `speed_hump_sign`, `zebra_crossing`. A zebra crossing is not a hazard, but labelling it as a separate class stops the model confusing it with painted bumps.

---

## 4. Recommended stack

### 4.1 Core

| Layer | Choice | Why / alternatives |
|---|---|---|
| Language | Python 3.11 | |
| DL framework | PyTorch 2.x | |
| Detector (start) | **Ultralytics YOLO11n/s**, or YOLO26 if available in your Ultralytics version | One-line train, val and export. Biggest community. ⚠️ **AGPL-3.0**: fine for school or open-source projects. A closed commercial app needs a paid licence. |
| Detector (licence-safe) | **RF-DETR** (Nano–Large, Apache-2.0) or **D-FINE** (Apache-2.0) | Similar or better accuracy, and allowed in commercial products. |
| Annotation | **CVAT** (free, self-host or cloud) or **Label Studio**. Roboflow if you want hosted annotation + augmentation + export | All export YOLO format. |
| Dataset QA | **FiftyOne** | Visualise, find duplicates and label mistakes, browse false positives and false negatives. |
| Augmentation | Built-in YOLO augmentations + **Albumentations** | Motion blur, rain, fog, shadow, low light. |
| Experiment tracking | **MLflow** (local) or **Weights & Biases** | Compare runs; keep the best weights. |
| Data versioning | **DVC** or plain zip snapshots + checksums | Keeps the dataset in sync with the git commit that trained on it. |
| Video and tracking | **OpenCV** + **supervision** (ByteTrack) | Stable track IDs across frames, which make alerts less noisy. |
| Demo UI | **Gradio** or **Streamlit** | Upload a video and see the detections. Good for presentations. |

### 4.2 Deployment

| Target | Export format | Runtime |
|---|---|---|
| Laptop / webcam | ONNX | ONNX Runtime (CPU/GPU) |
| NVIDIA Jetson | TensorRT engine (FP16/INT8) | TensorRT / DeepStream |
| Android | TFLite (INT8) or NCNN | TFLite + GPU delegate; NCNN with Vulkan |
| iOS | Core ML | Core ML / Vision |
| Raspberry Pi 5 (+ Hailo/Coral, optional) | NCNN / HEF / EdgeTPU TFLite | Low FPS without an accelerator |

### 4.3 Optional app / backend (for the map feature)

| Part | Choice |
|---|---|
| Mobile app | Kotlin (Android) + CameraX + TFLite, or Flutter with a TFLite plugin |
| API | FastAPI |
| Database | PostgreSQL + PostGIS (geo-queries such as "bumps within 500 m") |
| Map | Leaflet + OpenStreetMap tiles |

---

## 5. Development roadmap

### Phase 0: Setup (½ day)
- [ ] `git init` **inside** `speed-bump-detector/`. Right now the folder sits in a git repo rooted at `C:\`, which you almost certainly don't want.
- [ ] Create a virtual env (`uv` or `conda`), install `ultralytics fiftyone opencv-python albumentations`.
- [ ] Add a `.gitignore` for `data/`, `runs/`, `*.pt`, `*.onnx`. Store data with DVC or outside git.

### Phase 1: Data (the most important phase, 1–2 weeks)

**1.1 Clean**
- [ ] Remove the 76 duplicates from `Marked Hump/Bump/` (list in `dataset_audit_duplicates.csv`).
- [ ] Strip EXIF (GPS) and apply EXIF rotation.
- [ ] Look through every image in FiftyOne and delete blurry or ambiguous ones.

**1.2 Check public labelled sources**
- [ ] Roboflow *speed-bump-hump-marked*, *Indian Road Humps*, *spbumps*, *Road Safety AI – Rumble Strip, Speed Brakers*.
- [ ] Mendeley *Marked Speed Bump/Speed Breaker Dataset (India)* (969 cropped images) and *Manhole and Speed Breaker Detection in Bangladesh*.
- [ ] Merge only after mapping their class names to yours and re-checking their boxes.

**1.3 Annotate**
Labelling guideline (write it down and keep it consistent):
- Draw one box per bump, covering the **full visible width of the bump across the road** and its full depth.
- Label a bump if it's **≥ ~12 px tall** in the image. Skip smaller ones, or tag them `ignore`.
- If the bump is painted, it's `marked_bump`, even if the paint is faded but still visible. Otherwise it's `unmarked_bump`.
- Don't label zebra crossings or painted lines as bumps.
- **Speed-up:** label about 100 images by hand, train a quick YOLO11n, let it pre-label the rest, then **correct** its boxes. This is two to three times faster.

**1.4 Add negatives and hard negatives**
- [ ] Record **dashcam-style video** (phone on the windscreen at about 1.2 m, 30 FPS) on roads with and without bumps. Extract 1 frame/sec.
- [ ] Hard negatives to include: **zebra crossings, painted stop lines, shadows across the road, patched asphalt, manhole covers, rail crossings, road joints.**
- [ ] Optional: frames from the **RDD2022 India subset** (road damage, no bumps) as extra negative backgrounds.
- Target ratio: about **1:1 bump images to background images**.

**1.5 Split without leakage**
- Group images by **scene or burst** (filename sequence + perceptual-hash clustering, `imagehash` + simple clustering).
- Use **70 / 15 / 15** by group. All shots of one bump stay in one split.
- Keep a separate **"golden" test video**: a drive recorded *after* training starts and never used for tuning.

### Phase 2: Baseline (2–3 days)

```bash
yolo detect train model=yolo11n.pt data=data/bumps.yaml imgsz=640 epochs=150 \
     batch=16 patience=30 fliplr=0.5 flipud=0.0 degrees=0 mosaic=1.0 close_mosaic=10
```
- **Never flip vertically.** The road is always at the bottom of the frame. Horizontal flip is fine.
- Keep rotation small (≤ 5°) and use **perspective** augmentation, which looks like camera tilt.
- On a 4 GB GPU, use `batch=8` (or `batch=-1` for auto). Otherwise train on Colab/Kaggle.
- Record: mAP50, mAP50-95, **per-class recall**, FPS on target hardware.

### Phase 3: Improve (iterative, 1–3 weeks)

Do these roughly in order of payoff:
1. **Error analysis in FiftyOne**: sort by false negatives and false positives and look for patterns (far away? faded paint? shadows?).
2. **More and harder data**, focused on whatever kind of error dominates. Data beats architecture changes here.
3. **Bigger input** (`imgsz=960`) or a larger model (`11s` → `11m`) if distant bumps are missed. Bumps are wide but **very thin** in the image when far away.
4. **Road ROI crop**: run the model on the lower ~60 % of the frame, which gives more pixels per bump at the same speed.
5. **Domain augmentations** (Albumentations): `MotionBlur`, `RandomShadow`, `RandomRain`, `RandomFog`, `RandomBrightnessContrast`, low-light gamma.
6. **Pseudo-labelling** on your own unlabelled dashcam video, using only high-confidence predictions that you review by hand.
7. **Temporal logic** (see Phase 4). It often removes more false alarms than any model change.
8. Try **RF-DETR / D-FINE** and compare against YOLO on the same split.

### Phase 4: Real-time pipeline (1 week)

```
camera frame ──► resize/ROI ──► detector ──► tracker (ByteTrack)
                                                │
             GPS speed ──► distance + time-to-bump estimator ◄─┘
                                                │
                          alert logic (k-of-n frames, cooldown) ──► beep / voice / UI
```
- **Debounce:** alert only if the same track is detected in **≥ 3 of the last 5 frames**.
- **Cooldown:** one alert per bump, using the track ID and the GPS position.
- **Distance from a single camera:** with a fixed, calibrated camera (height *h*, pitch θ, focal *f*), the image row of the bump's bottom edge gives the ground distance through flat-road geometry. Calibrate once with a tape measure.
- **Time-to-bump** = distance / GPS speed. Warn at about 3–4 s.

### Phase 5: Deploy and field-test (1–2 weeks)
- Export: `yolo export model=best.pt format=tflite int8=True` (Android), `format=engine half=True` (Jetson), `format=onnx` (desktop).
- **Check INT8 accuracy** after quantisation against the golden test set.
- Drive real routes and log every alert. Compute the event-level metrics in §6.

---

## 6. Evaluation: what "good" means

mAP alone does not tell you whether the system is usable. Track these:

| Level | Metric | Suggested target (v1) |
|---|---|---|
| Model | mAP50 (overall / per class) | ≥ 0.85 marked, ≥ 0.65 unmarked |
| Model | Recall per class | ≥ 0.90 marked, ≥ 0.75 unmarked |
| Event | **Bump-level recall**: % of real bumps warned about at least *T* seconds before reaching them | ≥ 95 % (marked) at T ≥ 2 s |
| Event | **False alarms per 10 km** on roads without bumps | ≤ 1 |
| Event | Warning lead distance (median) | ≥ 25 m at 40 km/h |
| System | End-to-end latency | ≤ 100 ms |
| System | FPS on target device | ≥ 15 FPS (phone), ≥ 30 FPS (Jetson) |

Always report results on the **golden test video**, not only on the validation split.

---

## 7. Deployment targets

| Option | Pros | Cons | Good for |
|---|---|---|---|
| **Android phone app** | Everyone has one; it already has a camera, GPS and IMU | Thermal throttling, many different devices | Best demo for a student/portfolio project, and for crowdsourcing |
| **Jetson Orin Nano + USB/CSI cam** | Fast (TensorRT), stable | Costs about $250+, needs mounting and power | Vehicle prototype or research |
| **Laptop + webcam / recorded video** | Easiest | Not practical in a car | Development and presentation |
| **Raspberry Pi 5 + Hailo-8L** | Cheap and low power | More integration work | An embedded product prototype |

---

## 8. Features to add later

Grouped by value and effort. ⭐ = recommended next.

### 8.1 Driver-facing
| Feature | Idea | Effort |
|---|---|---|
| ⭐ **Distance & time-to-bump warning** | Flat-road geometry + GPS speed. Warn earlier at higher speed | Medium |
| ⭐ **Speed-adaptive alerts** | Alert only when the vehicle is above a safe crossing speed (e.g. > 20 km/h) | Low |
| Voice / haptic alerts | "Speed bump in 30 metres" | Low |
| Night & rain mode | Separate validation set for night and rain; low-light augmentation; IR camera if needed | Medium |
| **"Hump ahead" sign detection** | Second source of evidence; warns even when the bump is hidden by traffic | Medium |
| Pothole / rumble strip / manhole classes | Becomes a general **road hazard** detector (SBP-YOLO does bumps + potholes) | Medium |

### 8.2 Sensor fusion & mapping
| Feature | Idea | Effort |
|---|---|---|
| ⭐ **Accelerometer confirmation** | A vertical-acceleration spike confirms a detection. A spike with no detection means a missed bump | Medium |
| ⭐ **Automatic labels from the IMU** | When the IMU registers a jolt at time *t*, the frames from *t − 3 s … t − 1 s* contain the bump. Save them, so you collect labelled data just by driving | Medium |
| ⭐ **Shared bump map** | Upload geotagged detections; confirm a bump once **N different devices** report it (CRATER used > 5); warn from the map even before the camera sees it | High |
| Bump severity / height estimate | Size of the IMU spike vs speed → "mild / harsh" bump rating | Medium |
| Map-first warnings | Pre-download known bumps on the route and warn by GPS, so warnings work at night or in fog | Medium |

### 8.3 Public / municipal value
| Feature | Idea |
|---|---|
| **Unmarked-bump report** | Unmarked bumps are a safety issue. Export a list or map for the road authority with photo + GPS so they can be painted |
| Paint-condition monitoring | Classify `marked` vs `faded` paint to plan repainting |
| Dashboard | Heatmap of bumps and harsh-hit events per road segment |

### 8.4 Vehicle integration (advanced / research)
| Feature | Idea |
|---|---|
| Predictive / active suspension | Detect early → stiffen or soften the dampers before impact (SBP-YOLO's target use case) |
| ADAS speed advice | Feed bump distance into adaptive cruise control or a speed-advisory display |

---

## 9. What similar projects did

| Project / paper | Approach | Key result | What to take from it |
|---|---|---|---|
| **SBP-YOLO** (arXiv 2508.01339, 2025) | YOLO11n with a lightweight design; speed bumps + potholes; about 7,500 images merged from public sets | 87.0 % mAP, 139.5 FPS on Jetson AGX Xavier with TensorRT FP16 | Merging public datasets works; small YOLO + TensorRT is enough for real time |
| **SPD-YOLO** (J. Beijing Inst. Tech., 2025) | YOLO11s variant, speed bumps only | 79.9 % mAP, 26 % fewer params | Speed bumps are harder than generic objects; ~80 % mAP is a realistic ceiling on hard data |
| **FPNet** (Sensors, 2024) | FasterNet + Dynamic Snake Convolution for thin, elongated bumps | Large mAP gain over baseline | Bumps are long and thin; shape-aware features help |
| **Szabó et al.** (IEEE IPAS 2025) | YOLO + LiDAR for distance, Jetson AGX Xavier | Multi-hazard detection | Shows the distance-estimation step needed for warnings |
| **Marked Speed Breaker DAS** (Mendeley, India) | 969 cropped marked bumps | Dataset | Extra marked-bump data |
| **CRATER** (Univ. Birmingham) | Smartphone accelerometer crowdsensing → web map of bumps and potholes | Published only after > 5 users confirm | Model for the shared map + consensus rule |
| **Speed-breaker early warning** (USENIX NSDR 2012) | Phone accelerometer + GPS, shared warnings | Early mobile-sensing work | Detects bumps you haven't seen yet via the IMU |
| **Salman & Mian (2023)** | Deep learning on smartphone IMU data for bump characterisation | CNN > LSTM | Use a 1D-CNN for the IMU branch |
| **Roboflow Universe** projects | Many small labelled speed-bump sets (Indian Road Humps, spbumps, Road Safety AI…) | Hundreds of images each | Starting labels and extra data. Check licence and label quality |

---

## 10. Risks and pitfalls

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Data leakage** from bursts | Test scores look great, real-world results are poor | Group split + separate golden video |
| **Zebra crossings detected as marked bumps** | Main source of false alarms | Hard negatives, or a `zebra_crossing` class |
| **Unmarked bumps are barely visible** | Low recall even with a good model | More data, higher resolution, IMU fusion, shared map |
| **Distant bumps are only a few pixels tall** | Late warnings | ROI crop, `imgsz=960`, tiling (SAHI) only if FPS allows |
| **Domain shift** (camera height, country, weather) | Model breaks on new footage | Collect from the real mounting position; night and rain sets |
| **Licence (AGPL-3.0)** | Blocks a closed commercial app | Use RF-DETR / D-FINE, or buy an Ultralytics licence |
| **Privacy** (faces, plates, GPS) | Legal/ethical exposure if you publish data | Strip EXIF; blur faces and plates before sharing |
| **Driver distraction** | A noisy alert is worse than none | Debounce, cooldown, sound-only alerts |

---

## 11. Suggested repository layout

```
speed-bump-detector/
├── README.md
├── docs/
│   ├── PROJECT_GUIDE.md              ← this file
│   ├── dataset_audit_duplicates.csv  ← 76 cross-folder duplicates
│   └── labeling_guidelines.md        ← write in Phase 1
├── bumps-dataset/                    ← raw, read-only (DVC-tracked)
├── data/
│   ├── processed/{images,labels}/{train,val,test}/
│   └── bumps.yaml                    ← YOLO data config
├── src/
│   ├── data/      (clean.py, dedup.py, split_by_group.py, extract_frames.py)
│   ├── train/     (train.py, configs/)
│   ├── eval/      (event_metrics.py, golden_video_eval.py)
│   ├── inference/ (realtime.py, tracker.py, distance.py, alerts.py)
│   └── export/    (export.py)
├── apps/
│   ├── demo_gradio/
│   └── android/   (later)
├── notebooks/     (EDA, error analysis)
├── runs/          (gitignored)
└── requirements.txt / pyproject.toml
```

---

## 12. Next steps checklist

**Done (2026-11-19)**
- [x] Duplicates resolved, metadata stripped, scene-grouped split (`src/data/`).
- [x] `.gitignore`, `requirements.txt`, `labeling_guidelines.md`.
- [x] 297 images boxed (268 marked auto + review, 29 unmarked by eye); 597 RDD2022 negatives.

**This week**
- [ ] `git init` in the project folder (it currently sits in a repo rooted at `C:\`).
- [ ] Check the Roboflow *speed-bump-hump-marked* / *Indian Road Humps* projects in a browser for ready-made boxes.
- [ ] In CVAT, fix the 50 `needs_fix` and label the 120 `needs_label` images (`data/clean/label_review.csv`).

**Next week**
- [ ] Record about 30 minutes of dashcam-style video, with bumps and without.
- [ ] Pre-label with the first model, correct the boxes, and build the grouped 70/15/15 split.
- [ ] Train the YOLO11n baseline; record per-class recall and FPS.

**After that**
- [ ] Error analysis → targeted data collection → retrain (repeat).
- [ ] Real-time pipeline with tracker, debouncing and distance estimation.
- [ ] Export to the target device, then field-test with event-level metrics.
- [ ] Pick one feature from §8 (recommended: IMU confirmation + automatic labels).

---

## 13. References

**Papers**
- SBP-YOLO: lightweight real-time speed bump & pothole detection, arXiv 2508.01339 — https://arxiv.org/abs/2508.01339
- SPD-YOLO: lightweight YOLO for road information detection (J. Beijing Inst. Tech., 2025) — https://journal.hep.com.cn/jbit/EN/10.15918/j.jbit1004-0579.2025.054
- Szabó et al., Real-time detection of road hazards for autonomous vehicle systems (IEEE IPAS 2025) — https://eprints.sztaki.hu/10946/
- RDD2022: multi-national road damage dataset — https://arxiv.org/abs/2209.08538
- CRATER: crowd-sensing road conditions — https://research.birmingham.ac.uk/en/publications/crater-a-crowd-sensing-application-to-estimate-road-conditions-2
- Speed-breaker early warning system (USENIX NSDR 2012) — https://www.usenix.org/conference/nsdr12/speed-breaker-early-warning-system

**Datasets**
- Marked Speed Bump / Speed Breaker Dataset (India), Mendeley — https://data.mendeley.com/datasets/bvpt9xdjz8/1
- Manhole and Speed Breaker Detection (Bangladesh), Mendeley — https://data.mendeley.com/datasets/7zzcmv2wz7
- Roboflow: speed-bump-hump-marked — https://universe.roboflow.com/emirhan-ilhan/speed-bump-hump-marked
- Roboflow: Indian Road Humps — https://universe.roboflow.com/road-hump/indian-road-humps
- Roboflow: spbumps (Marked / Unmarked Bump) — https://universe.roboflow.com/sam-glt4b/spbumps
- Roboflow: Road Safety AI – Rumble Strip, Speed Brakers — https://universe.roboflow.com/guard-rails/road-safety-ai-rumble-strip-speed-brakers

**Tooling & licensing**
- Ultralytics docs (train / export) — https://docs.ultralytics.com
- Why AGPL-3.0 matters for CV teams (Roboflow blog; vendor source) — https://blog.roboflow.com/why-agpl-3-0-is-a-risk-for-computer-vision-teams/
- Ultralytics alternatives overview (Lightly) — https://www.lightly.ai/blog/best-ultralytics-alternatives-in-2026
- RF-DETR vs alternatives (Roboflow) — https://blog.roboflow.com/rf-detr-vs-alternatives/

> Figures quoted from papers come from abstracts and summaries. Check the full papers before citing them in a report.
