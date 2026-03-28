# ADR-002: GT Strategy and Priority Assessment
**Date**: 2026-03-28
**Status**: Accepted
**Participants**: Claude vs Codex (gpt-5.4), 3 rounds
**Debate Style**: constructive
**Confidence**: 8/10

## Context

OccAny Phase 2 completed (35 cases, F1=0.773). Before starting Phase 3, need to determine:
1. How far can camera-only 3D accident analysis go?
2. What GT is needed and how much?
3. What should be done first?

## Decision

### What OccAny Can and Cannot Do

**Can answer well (deliverable)**:
- ego vs observed (camera shock = physical evidence)
- coarse impact side (front / left / right, sector-level)
- near-field free-space collapse before impact
- counterpart approach sector at last visibility
- geometric evidence supporting/contradicting VLM narrative
- system confidence given observability

**Can answer partially (with caveats)**:
- collision type (coarsely, when visibility is decent)
- evasive maneuver or yaw shock detection
- approximate collision timing validation
- object class involved (if semantics stable enough)

**Cannot answer**:
- exact speed, TTC, impact angle, legal fault
- precise world-coordinate reconstruction across scenes
- hidden/off-camera impacts
- brake force, delta-v, driver intent, injury severity
- fine object trajectories without tracking layer

### GT Strategy: Three-Tier System

| Tier | Source | Count Target | Use |
|------|--------|-------------|-----|
| **Gold** | Human-reviewed (2-stage protocol) | 60-70 | Primary evaluation metric |
| **Silver** | Human-reviewed but ambiguous, or unconfirmed | ~90 | Exploratory analysis |
| **Bronze** | Strong machine agreement only | scalable | Triage/screening |

### Gold Review Protocol (80 videos → 60-70 Gold)

**Selection** (80 videos from existing 157):
- 20 easy agreement cases
- 20 VLM vs collision-type mismatch cases
- 20 accept_with_warning cases
- 20 category-balance (observed, side, night, edge cases)

**Stage 1 — Blind review**:
Reviewer sees raw video clip + collision-time marker only.
Records: accident_subject, impact_side, collision_time_ok, observability, confidence, rationale.

**Stage 2 — Assisted adjudication**:
Reveal VLM output + heuristic output + OccAny evidence (sector collapse, reliability, timestamp).
Reviewer may revise. Both pre-assist and final labels stored.

**Rules**:
- Gold = final human-adjudicated label
- Low confidence or poor observability → demote to Silver
- "Cannot tell" = valid outcome, do not force label

### Priority: Revised Sequence (consensus)

```
Step 1: Phase 3 Review-Assist Slice (3-4 days)
  └─ Dense windows + sectorized features + reliability + evidence report
  └─ Purpose: annotation instrument, NOT evaluated model yet

Step 2: Gold GT Sprint (2-3 days)
  └─ 80 videos human review using 2-stage protocol
  └─ Phase 3 evidence as review aid (not as GT source)
  └─ Freeze 60-70 Gold cases

Step 3: Phase 3 Formal Evaluation (2-3 days)
  └─ VLM claim verifier v1 on full 157
  └─ Primary metrics on Gold subset only
  └─ Failure taxonomy

Step 4: 1000+ Screening (1-2 days)
  └─ ego_signal as pre-filter on unlabeled accident pool
  └─ Output: likely_ego / likely_observed / uncertain
  └─ Conservative thresholds, triage only

Step 5: Targeted Label Expansion (as needed)
  └─ Only for underrepresented categories identified in Step 3
```

### 1000+ Accident Video Screening — MVP

For unlabeled videos beyond the 157:
1. Get candidate collision_time from VLM or risk-peak heuristic
2. Run OccAny only around `t ± 4s` (not full video)
3. Compute 4 outputs: ego_signal, best_sector, best_window_time, reliability
4. Bucket: likely_ego / likely_observed / uncertain

Output: `screening_results.json` — triage tool, not truth.

## Debate Summary

### Round 3 key insight (Codex, accepted)
> "Use Phase 3 first as an annotation instrument, then as an evaluated model."

This resolves the chicken-and-egg problem:
- Phase 3 outputs help human reviewer make better decisions
- Human decisions become Gold GT
- Gold GT evaluates Phase 3 formally
- No circularity because human is the labeler, model is evidence

### Claude position (maintained)
- Phase 3 fast track should come BEFORE GT sprint (as review tool)
- One human reviewer (the user) is acceptable for v1
- 1000+ screening is feasible as a practical screening tool with conservative thresholds

## Action Items
1. Build Phase 3 review-assist slice (dense windows + sectorized + reliability + evidence report)
2. Select 80 videos for Gold review
3. Create review tool (video + evidence side-by-side)
4. User reviews 80 videos with 2-stage protocol
5. Freeze Gold subset
6. Formal Phase 3 evaluation on Gold
7. Screen 1000+ pool for ego-involved candidates

## Explicitly NOT Doing
- 3D annotation or BEV trajectory GT
- Mass-labeling beyond 157 before evaluation proves value
- Auto-promoting machine agreement to Gold
- Running full video processing on 30K videos
- Training any ML model before rule-based validation

## Consequences
- Phase 3 becomes dual-purpose: review tool + evaluation target
- Gold subset enables credible per-category metrics
- 1000+ screening enables practical accident pool triage
- Deferred scale decisions until evaluation credibility is established
