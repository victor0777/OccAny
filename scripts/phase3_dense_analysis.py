"""
Phase 3 Fast Track: Dense collision-centered analysis with sectorized features.

For each video:
1. Extract frames at 5fps around collision_time ± 5s
2. Create overlapping 5-frame windows (stride 1)
3. Run OccAny on each window
4. Extract sectorized class-agnostic features per window
5. Compute reliability score
6. Generate evidence report
"""
import argparse
import json
import os
import sys
import subprocess
import shutil
import glob
import time
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCIDENT_DATA_DIR = "/data2/accident_data"


def extract_collision_frames(video_path, output_dir, collision_time, window_sec=5.0, fps=5):
    """Extract frames around collision_time at given fps."""
    os.makedirs(output_dir, exist_ok=True)
    start = max(0, collision_time - window_sec)
    duration = window_sec * 2
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
        "-t", str(duration), "-vf", f"fps={fps}", "-q:v", "2",
        os.path.join(output_dir, "%06d.jpg")
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    frames = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))
    return frames, start


def create_overlapping_windows(frames, scenes_dir, window_size=5, stride=1):
    """Create overlapping scene directories with stride-1 windows."""
    if os.path.exists(scenes_dir):
        shutil.rmtree(scenes_dir)
    windows = []
    for i in range(0, len(frames) - window_size + 1, stride):
        scene_dir = os.path.join(scenes_dir, f"window_{i:04d}")
        os.makedirs(scene_dir, exist_ok=True)
        for j in range(window_size):
            shutil.copy2(frames[i + j], scene_dir)
        windows.append(scene_dir)
    return windows


def run_occany(input_dir, output_dir, gpu_id=0):
    """Run OccAny inference."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(REPO_ROOT / "third_party" / "croco") + ":" + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, str(REPO_ROOT / "inference.py"),
        "--model", "occany_must3r", "--input_dir", input_dir,
        "--output_dir", output_dir, "--semantic", "pretrained@SAM2_large",
        "--compute_segmentation_masks", "--silent",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    return r.returncode == 0


def compute_sectorized_features(pts3d, conf, sem, poses, threshold=3.0):
    """Compute class-agnostic features per sector for one window.

    Sectors:
        front_left, front_center, front_right, left, right
    Based on image columns (horizontal position).
    """
    N, H, W = conf.shape
    mid = N // 2

    # Define sector column ranges
    sectors = {
        "left":         (0, W // 5),
        "front_left":   (W // 5, 2 * W // 5),
        "front_center": (2 * W // 5, 3 * W // 5),
        "front_right":  (3 * W // 5, 4 * W // 5),
        "right":        (4 * W // 5, W),
    }

    features = {}
    for sector_name, (col_start, col_end) in sectors.items():
        pts_sector = pts3d[mid, :, col_start:col_end]
        conf_sector = conf[mid, :, col_start:col_end]

        mask = conf_sector > threshold
        n_total = conf_sector.size
        n_valid = int(mask.sum())

        # Density: fraction of high-confidence points
        density = n_valid / n_total if n_total > 0 else 0.0

        # Min depth (5th percentile)
        if n_valid > 10:
            depths = np.linalg.norm(pts_sector[mask], axis=-1)
            min_depth = float(np.percentile(depths, 5))
            median_depth = float(np.median(depths))
        else:
            min_depth = float("nan")
            median_depth = float("nan")

        # Confidence stats
        conf_mean = float(conf_sector.mean())
        conf_std = float(conf_sector.std())

        features[sector_name] = {
            "density": density,
            "min_depth": min_depth,
            "median_depth": median_depth,
            "conf_mean": conf_mean,
            "conf_std": conf_std,
        }

    # Cross-frame density for density collapse detection
    per_frame_density = []
    for frame_idx in range(N):
        d = float((conf[frame_idx] > threshold).sum() / conf[frame_idx].size)
        per_frame_density.append(d)

    # Rotation shock: pose delta within window
    if poses is not None and len(poses) >= 2:
        # Rotation change between first and last frame
        R_first = poses[0, :3, :3]
        R_last = poses[-1, :3, :3]
        R_delta = R_last @ R_first.T
        # Rotation angle from trace
        trace = np.trace(R_delta)
        cos_angle = (trace - 1) / 2
        cos_angle = np.clip(cos_angle, -1, 1)
        rotation_shock = float(np.arccos(cos_angle))  # radians
    else:
        rotation_shock = 0.0

    # Free space ratio: fraction of low-confidence center
    center_conf = conf[mid, :, 2*W//5:3*W//5]
    free_space_ratio = float((center_conf < threshold).sum() / center_conf.size)

    features["_global"] = {
        "per_frame_density": per_frame_density,
        "rotation_shock": rotation_shock,
        "free_space_ratio": free_space_ratio,
        "mean_density": float(np.mean(per_frame_density)),
    }

    return features


def compute_window_features(window_output_dir, threshold=3.0):
    """Extract sectorized features from one window's OccAny output."""
    pts_path = os.path.join(window_output_dir, "pts3d_render.npy")
    vox_path = os.path.join(window_output_dir, "voxel_predictions.pkl")

    if not os.path.exists(pts_path):
        return None

    data = np.load(pts_path, allow_pickle=True).item()
    pts3d = data["pts3d"]
    conf = data["conf"]
    sem = data["semantic_2ds"]

    poses = None
    if os.path.exists(vox_path):
        with open(vox_path, "rb") as f:
            vox = pickle.load(f)
        poses = vox.get("estimated_input_camera_poses")

    return compute_sectorized_features(pts3d, conf, sem, poses, threshold)


def compute_reliability(window_features_list):
    """Compute per-window reliability score."""
    reliabilities = []
    for wf in window_features_list:
        if wf is None:
            reliabilities.append(0.0)
            continue
        g = wf["_global"]
        density = g["mean_density"]
        # Low density = low reliability
        # High free space = potentially unreliable
        # High rotation shock = potentially unreliable
        score = min(1.0, density * 3.0)  # scale: 0.33 density → 1.0 reliability
        if g["rotation_shock"] > 0.3:  # >17 degrees
            score *= 0.7
        if g["free_space_ratio"] > 0.8:
            score *= 0.5
        reliabilities.append(float(score))
    return reliabilities


def compute_density_collapse(window_features_list, reliabilities):
    """Compute per-sector density collapse across windows."""
    sectors = ["left", "front_left", "front_center", "front_right", "right"]
    sector_collapses = {}

    for sector in sectors:
        densities = []
        for wf in window_features_list:
            if wf is None:
                densities.append(float("nan"))
            else:
                densities.append(wf[sector]["density"])
        densities = np.array(densities)

        # Find max drop weighted by reliability
        valid = ~np.isnan(densities)
        if valid.sum() < 2:
            sector_collapses[sector] = {
                "max_drop": 0.0, "drop_window_idx": -1, "densities": densities.tolist()
            }
            continue

        drops = np.diff(densities)
        weighted_drops = drops.copy()
        for i in range(len(drops)):
            if i + 1 < len(reliabilities):
                weighted_drops[i] *= reliabilities[i + 1]

        min_idx = int(np.nanargmin(weighted_drops))
        max_drop = float(min(weighted_drops[min_idx], 0))

        sector_collapses[sector] = {
            "max_drop": abs(max_drop),
            "drop_window_idx": min_idx + 1,
            "densities": densities.tolist(),
        }

    # Best sector = highest collapse
    best_sector = max(sector_collapses, key=lambda s: sector_collapses[s]["max_drop"])
    return sector_collapses, best_sector


def generate_evidence_report(stem, collision_time, window_features_list,
                              reliabilities, sector_collapses, best_sector,
                              cause=None, subject=None, fps=5, start_time=0):
    """Generate structured evidence report for one video."""
    n_windows = len(window_features_list)
    n_valid = sum(1 for wf in window_features_list if wf is not None)

    # Global ego signal (max density collapse across all sectors)
    max_sector_drop = max(sc["max_drop"] for sc in sector_collapses.values())
    drop_window = sector_collapses[best_sector]["drop_window_idx"]
    impact_time_est = start_time + drop_window / fps if drop_window >= 0 else None

    # Rotation shocks
    rot_shocks = [wf["_global"]["rotation_shock"] if wf else 0 for wf in window_features_list]
    max_rot_idx = int(np.argmax(rot_shocks))
    max_rot = float(rot_shocks[max_rot_idx])

    # Asymmetry: left vs right collapse
    left_drop = sector_collapses["left"]["max_drop"] + sector_collapses["front_left"]["max_drop"]
    right_drop = sector_collapses["right"]["max_drop"] + sector_collapses["front_right"]["max_drop"]
    if left_drop + right_drop > 0:
        asymmetry = (left_drop - right_drop) / (left_drop + right_drop)
    else:
        asymmetry = 0.0

    # Depth approach in front_center
    fc_depths = [wf["front_center"]["min_depth"] if wf else float("nan") for wf in window_features_list]
    fc_valid = [d for d in fc_depths if not np.isnan(d)]
    approaching = False
    if len(fc_valid) >= 3:
        diffs = np.diff(fc_valid)
        approaching = bool(sum(d < -0.5 for d in diffs) >= 2)

    report = {
        "stem": stem,
        "collision_time_labeled": collision_time,
        "label_cause": cause,
        "label_subject": subject,
        "n_windows": n_windows,
        "n_valid_windows": n_valid,
        "mean_reliability": float(np.mean(reliabilities)),
        "evidence": {
            "max_sector_collapse": max_sector_drop,
            "best_collapse_sector": best_sector,
            "impact_time_estimated": impact_time_est,
            "impact_time_delta": abs(impact_time_est - collision_time) if impact_time_est else None,
            "max_rotation_shock_rad": max_rot,
            "max_rotation_shock_deg": float(np.degrees(max_rot)),
            "left_right_asymmetry": asymmetry,
            "approaching_object_front": approaching,
        },
        "sector_collapses": {s: {"max_drop": sc["max_drop"], "window_idx": sc["drop_window_idx"]}
                            for s, sc in sector_collapses.items()},
        "per_window_reliability": reliabilities,
        "per_window_rotation_shock": rot_shocks,
        "front_center_depths": fc_depths,
    }
    return report


def process_one_video(stem, collision_time, output_base, work_dir,
                       gpu_id=0, cause=None, subject=None):
    """Full Phase 3 pipeline for one video."""
    video_path = os.path.join(ACCIDENT_DATA_DIR, f"{stem}.mp4")
    result_path = os.path.join(output_base, f"{stem}_evidence.json")

    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    if not os.path.exists(video_path):
        return {"stem": stem, "error": "video_not_found"}

    frames_dir = os.path.join(work_dir, "frames")
    scenes_dir = os.path.join(work_dir, "scenes", stem)
    occany_out = os.path.join(work_dir, "occany_output", stem)

    # 1. Extract frames around collision
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    frames, start_time = extract_collision_frames(video_path, frames_dir, collision_time)
    if len(frames) < 5:
        return {"stem": stem, "error": "insufficient_frames"}

    # 2. Create overlapping windows
    windows = create_overlapping_windows(frames, scenes_dir, window_size=5, stride=1)
    if not windows:
        return {"stem": stem, "error": "no_windows"}

    # 3. Run OccAny
    success = run_occany(scenes_dir, occany_out, gpu_id=gpu_id)
    if not success:
        return {"stem": stem, "error": "inference_failed"}

    # 4. Extract sectorized features per window
    window_dirs = sorted(glob.glob(os.path.join(occany_out, "window_*_occany_must3r")))
    window_features = []
    for wd in window_dirs:
        wf = compute_window_features(wd)
        window_features.append(wf)

    if not any(wf is not None for wf in window_features):
        return {"stem": stem, "error": "no_valid_features"}

    # 5. Reliability
    reliabilities = compute_reliability(window_features)

    # 6. Sector collapse
    sector_collapses, best_sector = compute_density_collapse(window_features, reliabilities)

    # 7. Evidence report
    report = generate_evidence_report(
        stem, collision_time, window_features, reliabilities,
        sector_collapses, best_sector, cause=cause, subject=subject,
        fps=5, start_time=start_time,
    )

    os.makedirs(output_base, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(report, f, indent=2)

    # Cleanup
    shutil.rmtree(frames_dir, ignore_errors=True)

    return report


def main():
    parser = argparse.ArgumentParser(description="Phase 3 dense collision analysis")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "phase3_evidence"))
    parser.add_argument("--work-dir", default="/tmp/occany_phase3_work")
    args = parser.parse_args()

    with open("/home/ktl/projects/accident_analysis/collision_analysis_results.json") as f:
        coll = json.load(f)
    with open("/home/ktl/projects/accident_analysis/cause_classification_results.json") as f:
        causes = json.load(f)

    # Only daytime with collision_time
    stems = [k for k in coll if "_D_" in k and coll[k].get("collision_time")]
    if args.limit > 0:
        stems = stems[:args.limit]

    print(f"Phase 3: Processing {len(stems)} videos (GPU {args.gpu})")
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    errors = []
    for i, stem in enumerate(sorted(stems)):
        col_time = coll[stem]["collision_time"]
        subject = coll[stem].get("verified", {}).get("accident_subject", "?")
        cause = causes.get(stem, {}).get("cause", "?")

        print(f"[{i+1}/{len(stems)}] {stem} (t={col_time:.1f}s, cause={cause})")
        t0 = time.time()

        report = process_one_video(
            stem, col_time, args.output_dir, args.work_dir,
            gpu_id=args.gpu, cause=cause, subject=subject,
        )
        elapsed = time.time() - t0

        if "error" in report:
            print(f"  ERROR: {report['error']} ({elapsed:.1f}s)")
            errors.append(report)
        else:
            ev = report["evidence"]
            print(f"  sector={ev['best_collapse_sector']}, "
                  f"collapse={ev['max_sector_collapse']:.3f}, "
                  f"rot={ev['max_rotation_shock_deg']:.1f}deg, "
                  f"asym={ev['left_right_asymmetry']:.2f}, "
                  f"approach={ev['approaching_object_front']}, "
                  f"reliability={report['mean_reliability']:.2f} "
                  f"({elapsed:.1f}s)")
            results.append(report)

    with open(os.path.join(args.output_dir, "phase3_results.json"), "w") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2)

    print(f"\n=== Done: {len(results)} OK, {len(errors)} errors ===")


if __name__ == "__main__":
    main()
