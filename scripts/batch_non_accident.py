"""
Run OccAny on non_accident videos as control group.
Uses review_results.json (label=non_accident) instead of collision_analysis.
"""
import json
import os
import sys
import time
import shutil
import glob
import subprocess
import numpy as np
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCIDENT_DATA_DIR = "/data2/accident_data"
REVIEW_RESULTS = "/home/ktl/projects/accident_analysis/review_results.json"
FRAMES_PER_SCENE = 5
EXTRACT_FPS = 2
CONF_THRESHOLD = 3.0


def extract_frames(video_path, output_dir, fps=EXTRACT_FPS):
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2",
           os.path.join(output_dir, "%06d.jpg")]
    subprocess.run(cmd, capture_output=True, text=True)
    return len(glob.glob(os.path.join(output_dir, "*.jpg")))


def split_into_scenes(frames_dir, scenes_dir, chunk=FRAMES_PER_SCENE):
    if os.path.exists(scenes_dir):
        shutil.rmtree(scenes_dir)
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    sc = 0
    for i in range(0, len(frames), chunk):
        sd = os.path.join(scenes_dir, f"scene_{sc:04d}")
        os.makedirs(sd, exist_ok=True)
        for j in range(i, min(i + chunk, len(frames))):
            shutil.copy2(frames[j], sd)
        sc += 1
    return sc


def run_occany(input_dir, output_dir, gpu_id=0):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(REPO_ROOT / "third_party" / "croco") + ":" + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(REPO_ROOT / "inference.py"),
           "--model", "occany_must3r", "--input_dir", input_dir,
           "--output_dir", output_dir, "--semantic", "pretrained@SAM2_large",
           "--compute_segmentation_masks", "--silent"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    return r.returncode == 0


def analyze_results(output_dir):
    scene_dirs = sorted(glob.glob(os.path.join(output_dir, "scene_*_occany_must3r")))
    if not scene_dirs:
        return None
    densities = []
    min_depths = []
    for sd in scene_dirs:
        pts_path = os.path.join(sd, "pts3d_render.npy")
        if not os.path.exists(pts_path):
            continue
        data = np.load(pts_path, allow_pickle=True).item()
        conf = data["conf"]
        pts3d = data["pts3d"]
        N, H, W = conf.shape
        mid = N // 2
        densities.append(float((conf[mid] > CONF_THRESHOLD).sum() / conf[mid].size))
        center = slice(W // 3, 2 * W // 3)
        mask = conf[mid, :, center] > CONF_THRESHOLD
        if mask.sum() > 10:
            depths = np.linalg.norm(pts3d[mid, :, center][mask], axis=-1)
            min_depths.append(float(np.percentile(depths, 5)))
        else:
            min_depths.append(float("nan"))

    d_arr = np.array(densities)
    if len(d_arr) > 1:
        drops = np.diff(d_arr)
        max_drop = float(min(drops.min(), 0))
        ego_signal = abs(max_drop)
    else:
        ego_signal = 0.0

    return {
        "num_scenes": len(scene_dirs),
        "ego_signal_strength": ego_signal,
        "mean_density": float(np.mean(densities)) if densities else 0.0,
        "min_depth_overall": float(np.nanmin(min_depths)) if min_depths else float("nan"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "non_accident"))
    parser.add_argument("--work-dir", default="/tmp/occany_non_accident_work")
    args = parser.parse_args()

    with open(REVIEW_RESULTS) as f:
        review = json.load(f)

    stems = [k for k, v in review.items() if v.get("label") == "non_accident" and "_D_" in k]
    print(f"Processing {len(stems)} non_accident daytime videos (GPU {args.gpu})")

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    errors = []

    for i, stem in enumerate(sorted(stems)):
        video_path = os.path.join(ACCIDENT_DATA_DIR, f"{stem}.mp4")
        result_path = os.path.join(args.output_dir, stem, "analysis.json")

        if os.path.exists(result_path):
            with open(result_path) as f:
                results.append(json.load(f))
            print(f"[{i+1}/{len(stems)}] {stem} — cached")
            continue

        if not os.path.exists(video_path):
            errors.append({"stem": stem, "error": "not_found"})
            continue

        t0 = time.time()
        frames_dir = os.path.join(args.work_dir, "frames")
        scenes_dir = os.path.join(args.work_dir, "scenes", stem)
        out_dir = os.path.join(args.output_dir, stem)

        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        n = extract_frames(video_path, frames_dir)
        if n == 0:
            errors.append({"stem": stem, "error": "extract_failed"})
            continue

        split_into_scenes(frames_dir, scenes_dir)
        success = run_occany(scenes_dir, out_dir, gpu_id=args.gpu)
        if not success:
            errors.append({"stem": stem, "error": "inference_failed"})
            continue

        analysis = analyze_results(out_dir)
        if analysis is None:
            errors.append({"stem": stem, "error": "analysis_failed"})
            continue

        analysis["stem"] = stem
        analysis["label"] = "non_accident"
        os.makedirs(out_dir, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(analysis, f, indent=2)

        elapsed = time.time() - t0
        print(f"[{i+1}/{len(stems)}] {stem} — signal={analysis['ego_signal_strength']:.3f} ({elapsed:.1f}s)")
        results.append(analysis)
        shutil.rmtree(frames_dir, ignore_errors=True)

    with open(os.path.join(args.output_dir, "batch_results.json"), "w") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2)
    print(f"\n=== Done: {len(results)} OK, {len(errors)} errors ===")


if __name__ == "__main__":
    main()
