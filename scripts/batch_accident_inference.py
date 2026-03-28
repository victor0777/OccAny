"""
Batch OccAny inference on accident_analysis labeled videos.
Extracts frames, splits into 5-frame scenes, runs OccAny, and extracts 3D metrics.
"""
import argparse
import json
import os
import subprocess
import sys
import glob
import shutil
import time
import numpy as np
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCIDENT_DATA_DIR = "/data2/accident_data"
COLLISION_LABELS = "/home/ktl/projects/accident_analysis/collision_analysis_results.json"
COLLISION_TYPES = "/home/ktl/projects/accident_analysis/collision_types.json"
VLM_RESULTS = "/home/ktl/projects/accident_analysis/vlm_accident_results.json"

FRAMES_PER_SCENE = 5
EXTRACT_FPS = 2
CONF_THRESHOLD = 3.0


def load_labels():
    with open(COLLISION_LABELS) as f:
        collision = json.load(f)
    with open(COLLISION_TYPES) as f:
        ctypes = json.load(f)
    with open(VLM_RESULTS) as f:
        vlm = json.load(f)
    return collision, ctypes, vlm


def extract_frames(video_path, output_dir, fps=EXTRACT_FPS):
    """Extract frames from video at given fps."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}", "-q:v", "2",
        os.path.join(output_dir, "%06d.jpg")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0
    frames = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))
    return len(frames)


def split_into_scenes(frames_dir, scenes_dir, chunk=FRAMES_PER_SCENE):
    """Split extracted frames into scene directories of `chunk` frames each."""
    if os.path.exists(scenes_dir):
        shutil.rmtree(scenes_dir)

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    scene_count = 0
    for i in range(0, len(frames), chunk):
        scene_dir = os.path.join(scenes_dir, f"scene_{scene_count:04d}")
        os.makedirs(scene_dir, exist_ok=True)
        for j in range(i, min(i + chunk, len(frames))):
            shutil.copy2(frames[j], scene_dir)
        scene_count += 1
    return scene_count


def run_occany_inference(input_dir, output_dir, gpu_id=0):
    """Run OccAny Must3R + SAM2 inference."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(REPO_ROOT / "third_party" / "croco") + ":" + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable, str(REPO_ROOT / "inference.py"),
        "--model", "occany_must3r",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--semantic", "pretrained@SAM2_large",
        "--compute_segmentation_masks",
        "--silent",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    return result.returncode == 0


def analyze_scene(scene_output_dir):
    """Extract 3D metrics from one scene's OccAny output."""
    pts_path = os.path.join(scene_output_dir, "pts3d_render.npy")
    vox_path = os.path.join(scene_output_dir, "voxel_predictions.pkl")

    if not os.path.exists(pts_path) or not os.path.exists(vox_path):
        return None

    data = np.load(pts_path, allow_pickle=True).item()
    with open(vox_path, "rb") as f:
        vox = pickle.load(f)

    pts3d = data["pts3d"]       # (N, H, W, 3)
    conf = data["conf"]         # (N, H, W)
    sem = data["semantic_2ds"]  # (N, H, W)
    poses = vox["estimated_input_camera_poses"]  # (N, 4, 4)

    N, H, W = conf.shape
    mid = N // 2

    # 1. Min depth (center FOV, 5th percentile)
    center_strip = slice(W // 3, 2 * W // 3)
    pts_center = pts3d[mid, :, center_strip]
    conf_center = conf[mid, :, center_strip]
    mask = conf_center > CONF_THRESHOLD
    if mask.sum() > 10:
        depths = np.linalg.norm(pts_center[mask], axis=-1)
        min_depth = float(np.percentile(depths, 5))
        median_depth = float(np.median(depths))
    else:
        min_depth = float("nan")
        median_depth = float("nan")

    # 2. Reconstruction quality (high-conf ratio)
    density = float((conf[mid] > CONF_THRESHOLD).sum() / conf[mid].size)

    # 3. Camera position
    cam_pos = poses[mid, :3, 3].tolist()

    # 4. Spatial extent
    pts_valid = pts3d[mid][conf[mid] > CONF_THRESHOLD]
    if len(pts_valid) > 10:
        extent = (pts_valid.max(axis=0) - pts_valid.min(axis=0)).tolist()
    else:
        extent = [0.0, 0.0, 0.0]

    # 5. Semantic composition
    sem_flat = sem[mid].flatten()
    unique, counts = np.unique(sem_flat, return_counts=True)
    sem_ratios = {int(u): float(c / sem_flat.size) for u, c in zip(unique, counts)}

    return {
        "min_depth": min_depth,
        "median_depth": median_depth,
        "density": density,
        "cam_pos": cam_pos,
        "spatial_extent": extent,
        "semantic_ratios": sem_ratios,
    }


def analyze_video_results(output_dir, collision_time=None):
    """Analyze all scenes for one video and compute aggregate metrics."""
    scene_dirs = sorted(glob.glob(os.path.join(output_dir, "scene_*_occany_must3r")))
    if not scene_dirs:
        return None

    timeline = []
    for si, sd in enumerate(scene_dirs):
        metrics = analyze_scene(sd)
        if metrics is None:
            continue
        metrics["scene_idx"] = si
        metrics["time_sec"] = si * (FRAMES_PER_SCENE / EXTRACT_FPS)
        timeline.append(metrics)

    if not timeline:
        return None

    # Aggregate metrics
    densities = [t["density"] for t in timeline]
    min_depths = [t["min_depth"] for t in timeline]
    cam_positions = np.array([t["cam_pos"] for t in timeline])

    # Velocity from consecutive poses
    velocities = [0.0]
    dt = FRAMES_PER_SCENE / EXTRACT_FPS
    for i in range(1, len(cam_positions)):
        v = float(np.linalg.norm(cam_positions[i] - cam_positions[i - 1]) / dt)
        velocities.append(v)

    # Impact detection: density drop
    density_arr = np.array(densities)
    if len(density_arr) > 1:
        density_drops = np.diff(density_arr)
        max_drop_idx = int(np.argmin(density_drops))
        max_drop_val = float(density_drops[max_drop_idx])
        impact_time_density = timeline[max_drop_idx + 1]["time_sec"]
    else:
        max_drop_val = 0.0
        impact_time_density = float("nan")

    # Impact detection: velocity spike
    vel_arr = np.array(velocities)
    max_vel_idx = int(np.argmax(vel_arr))
    impact_time_velocity = timeline[max_vel_idx]["time_sec"]

    # Min distance before collision
    valid_depths = [(t["time_sec"], t["min_depth"]) for t in timeline if not np.isnan(t["min_depth"])]

    # Ego vs observed signal: large density drop = ego impact
    # (observed accidents keep stable reconstruction)
    ego_signal_strength = abs(max_drop_val) if max_drop_val < 0 else 0.0

    result = {
        "num_scenes": len(scene_dirs),
        "num_analyzed": len(timeline),
        "timeline": timeline,
        "velocities": velocities,
        "impact_time_from_density": impact_time_density,
        "impact_time_from_velocity": impact_time_velocity,
        "max_density_drop": max_drop_val,
        "ego_signal_strength": ego_signal_strength,
        "min_depth_overall": float(np.nanmin(min_depths)) if min_depths else float("nan"),
        "mean_density": float(np.mean(densities)),
    }

    # TTC estimation (if depth is decreasing)
    if len(valid_depths) >= 2:
        depths_arr = np.array([d for _, d in valid_depths])
        times_arr = np.array([t for t, _ in valid_depths])
        depth_rate = np.diff(depths_arr) / np.diff(times_arr)
        approaching = depth_rate < -0.1
        if approaching.any():
            idx = np.where(approaching)[0][-1]
            ttc = -depths_arr[idx + 1] / depth_rate[idx]
            result["ttc_estimate"] = float(ttc)
        else:
            result["ttc_estimate"] = float("nan")
    else:
        result["ttc_estimate"] = float("nan")

    return result


def process_one_video(stem, work_dir, output_base, gpu_id=0):
    """Full pipeline for one video: extract → split → infer → analyze."""
    video_path = os.path.join(ACCIDENT_DATA_DIR, f"{stem}.mp4")
    if not os.path.exists(video_path):
        return {"stem": stem, "error": "video_not_found"}

    frames_dir = os.path.join(work_dir, "frames")
    scenes_dir = os.path.join(work_dir, "scenes", stem)
    output_dir = os.path.join(output_base, stem)

    # Skip if already processed
    result_path = os.path.join(output_dir, "analysis.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    # Extract frames
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    n_frames = extract_frames(video_path, frames_dir)
    if n_frames == 0:
        return {"stem": stem, "error": "frame_extraction_failed"}

    # Split into scenes
    n_scenes = split_into_scenes(frames_dir, scenes_dir)

    # Run OccAny
    success = run_occany_inference(scenes_dir, output_dir, gpu_id=gpu_id)
    if not success:
        return {"stem": stem, "error": "inference_failed", "n_frames": n_frames}

    # Analyze
    analysis = analyze_video_results(output_dir)
    if analysis is None:
        return {"stem": stem, "error": "analysis_failed", "n_frames": n_frames}

    analysis["stem"] = stem
    analysis["n_frames"] = n_frames
    analysis["n_scenes"] = n_scenes

    # Save per-video result
    os.makedirs(output_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # Cleanup frames (keep scenes output)
    shutil.rmtree(frames_dir, ignore_errors=True)

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Batch OccAny accident analysis")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Process only N videos (0=all)")
    parser.add_argument("--daytime-only", action="store_true", help="Only daytime videos")
    parser.add_argument("--warnings-only", action="store_true", help="Only accept_with_warning cases")
    parser.add_argument("--work-dir", default="/tmp/occany_accident_work")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "accident_analysis"))
    args = parser.parse_args()

    collision, ctypes, vlm = load_labels()

    # Build video list
    stems = list(collision.keys())

    if args.daytime_only:
        stems = [s for s in stems if "_D_" in s]

    if args.warnings_only:
        stems = [s for s in stems if collision[s].get("warnings")]

    if args.limit > 0:
        stems = stems[:args.limit]

    print(f"Processing {len(stems)} videos (GPU {args.gpu})")
    print(f"Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    errors = []

    for i, stem in enumerate(stems):
        label = collision[stem]
        ctype = ctypes.get(stem, {}).get("type", "unknown")
        subject = label.get("verified", {}).get("accident_subject", "unknown")
        col_time = label.get("collision_time", None)

        print(f"\n[{i+1}/{len(stems)}] {stem} (subject={subject}, type={ctype}, t={col_time}s)")
        t0 = time.time()

        result = process_one_video(stem, args.work_dir, args.output_dir, gpu_id=args.gpu)

        elapsed = time.time() - t0

        if "error" in result:
            print(f"  ERROR: {result['error']} ({elapsed:.1f}s)")
            errors.append(result)
        else:
            # Add labels to result
            result["label_subject"] = subject
            result["label_type"] = ctype
            result["label_collision_time"] = col_time
            result["has_warning"] = bool(label.get("warnings"))
            print(f"  OK: {result['num_scenes']} scenes, "
                  f"ego_signal={result['ego_signal_strength']:.3f}, "
                  f"min_depth={result['min_depth_overall']:.1f}m, "
                  f"ttc={result.get('ttc_estimate', float('nan')):.1f}s "
                  f"({elapsed:.1f}s)")
            all_results.append(result)

    # Save aggregate results
    summary_path = os.path.join(args.output_dir, "batch_results.json")
    with open(summary_path, "w") as f:
        json.dump({"results": all_results, "errors": errors}, f, indent=2)
    print(f"\n=== Done: {len(all_results)} OK, {len(errors)} errors ===")
    print(f"Results: {summary_path}")


if __name__ == "__main__":
    main()
