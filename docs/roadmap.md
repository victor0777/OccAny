# Roadmap

## Phase 1: 환경 구축 + 데이터 호환성 검증 ✅
- [x] OccAny inference 환경 구축 (체크포인트, 의존성, 패치)
- [x] RTB camera_front 데이터 추론 성공 (5프레임, 595프레임)
- [x] 사고 영상 (야간/주간) 추론 성공
- [x] 3D 분석 지표 추출 파이프라인 프로토타입

## Phase 2: accident_analysis 대규모 분석 ✅
- [x] 배치 추론 스크립트 구축 (batch_accident_inference.py)
- [x] warning+daytime 35건 추론 완료 (에러 0건, 총 24분)
- [x] ego/observed 판별: ego_signal_strength F1=0.773 (threshold=0.08)
- [ ] 전체 157건 추론 + 통계적 유의성 확보
- [ ] ego/observed 재판정 결과 → accident_analysis에 피드백

## Phase 3: 3D Evidence Layer — Fast Track ← **current**
> ADR-001 (Claude vs Codex debate). OccAny = "3D geometric evidence layer", not standalone analyzer.
> Trust: scale-free relative features. Caution: absolute kinematics.

- [ ] Collision-centered dense windows (stride 1, collision_time ± 5s) — 1-2d
- [ ] Sectorized class-agnostic features (front-L/C/R, left, right) — 1-2d
- [ ] Reliability gating (per-window observability score) — 1d
- [ ] VLM claim verifier v1: rule-based, 4 claim types — 2-3d
  - ego_vs_observed, impact_side, counterpart_location, approaching_object
- [ ] Evaluate on 157 labeled + failure taxonomy — 1-2d

## Phase 4: Object-Centric Track
- [ ] Save SAM2 instance mask IDs (not just semantic_2ds) — 1-2d
- [ ] Anonymous object-centric 3D lifting (centroid, sector, approach) — 2-4d
- [ ] Semantic consistency MVP (voxel-level class voting within 5-frame scene) — 2-3d

## Phase 5: Scale & Integration
- [ ] 전체 157건 Phase 3 파이프라인 적용 + 30K coarse-to-fine 전략
- [ ] accident_analysis VLM 재판정 피드백 루프
- [ ] Mask2Former stable class prior + OccAny geometry 융합
