# Labelling Guidelines: Speed Bumps

Use these rules for every box (draft, review or new data) so that labels stay consistent.

## Classes

| id | name | Use when |
|---:|---|---|
| 0 | `marked_bump` | The bump has paint on it (zebra, chevrons, black/yellow or black/white blocks), even if the paint is faded but still visible. |
| 1 | `unmarked_bump` | A raised bump with no paint. Usually a darker or lighter band across the road, a ridge or a shadow line. |

## What a box covers

- **One box per bump**, covering the **full visible width of the bump across the road** (kerb to kerb, or the edge of the paint) and its **full depth**, from the near edge to the far edge.
- If a vehicle or person hides part of the bump, box the visible part plus the hidden part you can confidently infer between the visible ends.
- Label every bump in the image. A second bump further down the road gets its own box.
- **Minimum size:** skip bumps less than about **12 px tall** in the original image. They are too small to learn from and too far away to matter.

## What is NOT a bump (never box these)

| Looks like a bump | Why it isn't |
|---|---|
| **Zebra pedestrian crossings** (wide white bars along the direction of travel, flat) | Flat paint, no hump. Several images in `Unmarked bumps` (`u_1`, `u_2`, `u_3`, `u_8`) contain one next to the real bump. |
| Stop lines, lane arrows, centre lines | Flat paint |
| Shadows of poles, wires or trees across the road | Not a physical bump |
| Patched asphalt, manholes, road joints, potholes | Different hazard |

## Ambiguous images

- If a photo in the bump folders shows **no visible bump** (some unmarked approach shots), leave its label file **empty** and set `exclude=1` in the manifest. Don't guess.
- If you can't decide between marked and unmarked, use `marked_bump` when any paint is visible.

## Review workflow

1. Draft boxes come from `src/data/autolabel.py` (Grounding DINO, zero-shot) and from the manual corrections in `data/clean/label_review.csv`.
2. Import `data/yolo` into **CVAT** (Import → "YOLO 1.1") or **Label Studio**. Fix boxes, then export back to YOLO format.
3. Each row of `data/clean/label_review.csv` has a status: `ok` (draft accepted as is), `fixed` (box redrawn by hand) or `excluded` (no usable bump). Rows marked `fixed` are rough by-eye boxes, so check them first.
