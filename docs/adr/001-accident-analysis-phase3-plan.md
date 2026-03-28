# ADR-001: OccAny Accident Analysis Phase 3 Plan
**Date**: 2026-03-28
**Status**: Accepted
**Participants**: Claude vs Codex (gpt-5.4)
**Debate Style**: constructive
**Confidence**: 7/10

## Context

OccAny (camera-only 3D occupancy prediction) has been validated on 35 accident videos with ego_signal_strength achieving F1=0.773 for ego vs observed classification. The question is: what should Phase 3 look like to maximize value from this 3D evidence layer?

## Decision

### Core Principle (consensus)
**OccAny should be treated as a "3D geometric evidence layer" that verifies/corrects VLM narratives, NOT as a standalone accident analyzer.** The strongest signals are scale-free relative features (density drops, sector asymmetry), not absolute kinematics (velocity, TTC in meters/seconds).

### What to trust (no calibration available)
- **Trust**: left/right/front asymmetry, near/far ordering, free-space collapse, within-scene rotation shock, sign of approach, post-impact blockage
- **Caution**: absolute meters, absolute speed, TTC in seconds, cross-scene translation magnitude

### Phase 3 Task List (prioritized)

#### Fast Track (highest value, ~1 week)

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 1 | **Collision-centered dense windows** | 1-2 days | Very high |
|   | 5-frame windows with stride 1, only around `collision_time ± 4-6s` | | |
|   | Output: `window_features.jsonl` per video | | |
| 2 | **Sectorized class-agnostic features** | 1-2 days | Very high |
|   | Front-L/C/R, Left, Right sectors. Per window: density, min-depth, conf collapse, free-space ratio, rotation shock | | |
|   | Class-agnostic first (avoids SAM2 instability) | | |
| 3 | **Reliability gating** | 1 day | High |
|   | Per-window observability score from density, conf variance, empty-mask rate | | |
|   | Downweight low-trust windows instead of forcing prediction | | |
| 4 | **VLM claim verifier v1** | 2-3 days | High |
|   | Rule-based (no ML). 4 claim types: ego_vs_observed, impact_side, counterpart_location, approaching_object_present | | |
|   | Deterministic scores for easy failure analysis | | |

#### Object-Centric Track (after fast track, ~1-2 weeks)

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 5 | **Save SAM2 instance mask IDs** | 1-2 days | Medium-high |
|   | Currently only `semantic_2ds` saved, not per-object IDs | | |
| 6 | **Anonymous object-centric 3D lifting** | 2-4 days | High |
|   | Per SAM2 object: 3D centroid, sector, depth band, area, approach sign | | |
|   | Short tracklets within 5-frame scenes | | |
| 7 | **Semantic consistency MVP** | 2-3 days | Medium-high |
|   | Voxel-level class histogram voting within each 5-frame scene | | |
|   | Confidence-weighted, margin threshold for unknown | | |

#### Validation

| # | Task | Effort |
|---|------|--------|
| 8 | **Evaluate on 157 labeled videos + failure taxonomy** | 1-2 days |

### VLM Claim Verifier v1 — Architecture

**Input**:
- `vlm_result.json`: summary, claims, confidence, collision_time
- `window_features.jsonl`: sectorized density/depth/conf, rotation shock, free-space collapse, reliability

**4 Claim Types**:
```
ego_vs_observed   → sector collapse magnitude + reliability
impact_side       → sector collapse asymmetry + yaw shock sign
counterpart_location → near-field closure by sector
approaching_object  → monotonic near-field approach in any sector
```

**Output**: `claim_verification.json` with per-claim support_score, evidence dict, overall consistent/mixed/contradicted

### 3D Semantic Consistency MVP — Algorithm

1. Keep points with conf > threshold
2. Quantize to coarse voxel grid in shared scene frame
3. Per voxel: accumulate class histogram from all 5 views
4. Weight votes by point confidence
5. Assign top class only if margin > threshold, else `unknown`
6. Repaint per-pixel labels from voxel lookup

## Debate Summary

### Key Codex challenges (accepted)
- Cross-scene pose comparability is broken → velocity/TTC across chunks unreliable
- 2 FPS with non-overlapping chunks too sparse for crash dynamics → dense windows near collision_time
- ego_signal_strength works because it's scale-free → keep designing scale-free features
- Don't use logistic regression yet → deterministic rules first for debuggability
- SAM2 already runs in video mode → class instability is inherent, not a mode issue

### Key Claude positions (maintained)
- 5-frame OOM limit is hard constraint → no workaround without model change
- Running vehicle-tracker on 30K videos is out of scope → SAM2 instance masks as lighter path
- Phase 3 fast track should deliver value in 1 week

## Action Items
1. Implement collision-centered dense windows
2. Implement sectorized class-agnostic features
3. Add reliability gating
4. Build VLM claim verifier v1 (rule-based)
5. Run on 157 labeled videos, write failure taxonomy
6. Then decide whether to invest in object-centric track

## Explicitly NOT Doing
- Training/fine-tuning OccAny (inference only)
- Running vehicle-tracker on accident videos (too expensive)
- Cross-scene pose alignment (fundamentally unreliable without calibration)
- ML-based classifier (premature before rule-based validation)
- Scaling to 30K unlabeled videos (not until labeled evaluation is solid)

## Consequences
- Fast track delivers ego/observed + impact_side verification in ~1 week
- Rule-based approach makes failures transparent and debuggable
- Deferred object-centric work until fast track is validated
- Sectorized features may enable collision type reclassification (58 mismatches)
