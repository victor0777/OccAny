# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OccAny is a unified framework for generalized unconstrained urban 3D occupancy prediction (CVPR 2026) by Valeo.ai. It provides inference code and pretrained checkpoints for predicting 3D occupancy grids from RGB images. Two model variants exist: **OccAny** (Must3R + SAM2) and **OccAny+** (Depth Anything 3 + SAM3).

Training code, evaluation scripts, and dataset preparation are not yet released (TODO).

## Git Repository

- **Origin**: https://github.com/valeoai/OccAny.git (upstream, read-only fork from valeoai)

## Setup & Installation

```bash
conda create -n occany python=3.12
conda activate occany
pip install torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
# torch-scatter requires CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.6
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
# Optional: compile CroCo RoPE extension
cd third_party/croco/models/curope && python setup.py build_ext --inplace
```

Checkpoints are downloaded from HuggingFace (`anhquancao/OccAny`).

## Running Inference

```bash
# OccAny+ (DA3 + SAM3)
python inference.py --model occany_plus.pth --input_dir demo_data/input --output_dir demo_data/output \
  --gen -vpi 3 -rot 60 --semantic sam3@sam3_hiera_large.pt --compute_segmentation_masks

# OccAny (Must3R + SAM2)
python inference.py --model occany.pth --input_dir demo_data/input --output_dir demo_data/output \
  --gen -vpi 3 -rot 60 --semantic sam2@sam2.1_hiera_large.pt --compute_segmentation_masks
```

## Visualization

```bash
# Interactive point cloud viewer (viser)
python vis_viser.py --input_folder ./demo_data/output

# Voxel grid rendering (mayavi)
python vis_voxel.py --input_root ./demo_data/output --dataset nuscenes
```

## Architecture

### Data Flow
```
Input RGB → Model (Must3R or DA3) → Feature Extraction + Pose Estimation
  → [Optional] Novel View Generation → [Optional] Semantic Segmentation (SAM2/SAM3)
  → Point Cloud Lifting → Voxel Grid Fusion (TSDF) → [Optional] Majority Pooling
  → Output: pts3d_*.npy + voxel_predictions.pkl
```

### Key Modules

- **`inference.py`** — Main entry point. Orchestrates the full pipeline with ~95 CLI arguments covering model selection, generation, semantic segmentation, and post-processing.

- **`occany/model/`** — Model definitions:
  - `model_must3r.py` — Must3R backbone (`Dust3rEncoder`, `Must3r`, `OccAnyMust3r`). Extends Dust3R with causal decoding for multi-view 3D reconstruction.
  - `model_da3.py` — Depth Anything 3 wrapper (`DA3Wrapper`). Extends DA3 with raymap-conditioned generation.
  - `model_sam2.py` / `sam3_model.py` — SAM2/SAM3 segmentation wrappers for semantic labeling.
  - `occany_head.py` — Output heads for occupancy prediction.
  - `must3r_blocks/` — Must3R-specific transformer blocks (attention, layers, positional embeddings).

- **`occany/` inference modules**:
  - `must3r_inference.py` — Must3R-specific dense matching and reconstruction.
  - `da3_inference.py` — DA3 inference with novel view generation support (`inference_occany_da3()`, `inference_occany_da3_gen()`).
  - `semantic_inference.py` — Singleton `ModelManager` for SAM2/SAM3 feature extraction and semantic segmentation.

- **`occany/utils/`** — Core utilities:
  - `helpers.py` — Geometric transforms, voxelization (`voxelize_points()`, `create_voxel_prediction()`), pose generation (`generate_intermediate_poses()`), majority pooling.
  - `fusion.py` — `TSDFVolume` class for volumetric TSDF fusion.
  - `inference_helper.py` — Input parsing, resolution handling, format conversion between DA3 and OccAny.
  - `image_util.py` — Image normalization (`ImgNorm`), SAM2/SAM3 transforms.

### Third-Party Dependencies (vendored in `third_party/`)

- **dust3r/** — 3D reconstruction base (Dust3R)
- **croco/** — CroCo backbone with RoPE positional encoding (has a compilable `curope` C extension)
- **Grounded-SAM-2/** — SAM2 + GroundingDINO for open-vocabulary segmentation
- **sam3/** — SAM3 segmentation model
- **Depth-Anything-3/** — Monocular depth estimation

These are imported via relative paths from `inference.py` (sys.path manipulation). No package installation needed.

### Voxel Grid Presets

| Dataset   | voxel_size | occ_size         | voxel_origin          |
|-----------|-----------|------------------|-----------------------|
| Default   | 0.4       | [200, 200, 24]   | [-40, -40, -3.6]      |
| KITTI     | 0.2       | [256, 256, 32]   | [0, -25.6, -2.0]      |
| nuScenes  | 0.4       | [200, 200, 16]   | [-40, -40, -1.0]      |

### Output Format

- `pts3d_render.npy` / `pts3d_render_gen.npy` — Reconstructed/generated point clouds
- `voxel_predictions.pkl` — Semantic voxel grid predictions (pickled dict with grid + metadata)
