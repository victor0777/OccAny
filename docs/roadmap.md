# Roadmap

## Phase 1: 환경 구축 + 데이터 호환성 검증 ✅
- [x] OccAny inference 환경 구축 (체크포인트, 의존성, 패치)
- [x] RTB camera_front 데이터 추론 성공 (5프레임, 595프레임)
- [x] 사고 영상 (야간/주간) 추론 성공
- [x] 3D 분석 지표 추출 파이프라인 프로토타입

## Phase 2: accident_analysis 대규모 분석 ✅
- [x] 배치 추론 스크립트 구축 (batch_accident_inference.py)
- [x] 주간 121건 + non_accident 26건 = 147건 추론 완료
- [x] ego_signal 한계 확인: ego/observed F1=0.642, accident/non_accident 구분 불가
- [x] 10개 feature 식별 + 사고 원인 8종별 패턴 분석
- [x] dgx01 데이터 통합 (cause_classification, risk_events, physics_sim, gt_alignment)

## Phase 3: 3D Evidence Layer — Fast Track ← **current** (진행중)
> ADR-001 (Claude vs Codex debate). OccAny = "3D geometric evidence layer"

- [x] Phase 3 dense analysis 스크립트 구축 (phase3_dense_analysis.py)
- [ ] 121건 dense window 분석 완료 대기
- [ ] Sectorized features 통계 분석 + ego_signal 대비 개선 확인
- [ ] VLM claim verifier v1: rule-based, 4 claim types — 2-3d
- [ ] Evaluate on 157 labeled + failure taxonomy — 1-2d

## Phase 4: Gold GT Sprint
> ADR-002. GT schema v2: per-collision, dual contact zone, lane info

- [ ] GT schema v2 리뷰 도구 구축 (영상 + OccAny evidence 병렬 표시)
- [ ] 기존 L4 (15건) + expert_review (20건) 병합 → 시작점 ~35건
- [ ] 추가 45건 선정 + human review (2-stage: blind → assisted)
- [ ] 80건 Gold GT 확정 (per-collision, lane info 포함)
- [ ] 카테고리 밸런스 확인 (ego_passive, intersection 부족 시 unknown 500건에서 보충)

## Phase 5: Object-Centric + Lane Detection
> 사고 원인 분석의 핵심 gap 해소

### 5-1. 차량 추적 (Vehicle Tracking)
- [ ] SAM2 instance mask ID 저장 (현재 semantic_2ds만)
- [ ] 5프레임 내 anonymous object tracking (3D centroid 기반)
- [ ] 또는 YOLO+BYTETrack을 사고 영상에 적용 (별도 파이프라인)
- 활용: cut_in(11건), observed(58건) = 44% 사고에 필요

### 5-2. 차선 인식 (Lane Detection)
- [ ] VP EgoLanes 출력에서 차선 수/위치 추정
- [ ] 또는 CLRerNet을 사고 영상에 적용 (perception 프로젝트 연계)
- [ ] OccAny road 시맨틱에서 차선 마크 감지 가능성 탐색
- 활용: unsafe_lane_change(14건), cut_in(11건), solo(6건) = 20% 사고에 필요

### 5-3. 차선 대비 차량 위치 (Lane-relative Position)
- [ ] 5-1 + 5-2 결합: "car1이 2차선 → 1차선 이동 후 ego와 충돌"
- [ ] GT lane_a / lane_b 검증

## Phase 6: ML 전환
> 전제: Gold GT 60+건, 룰 기반 baseline F1 확인, 카테고리당 30+건

- [ ] 10개 OccAny feature + VP feature 결합 → feature matrix
- [ ] Logistic regression / gradient boosting 학습
- [ ] Gold로 평가, Silver로 보조
- [ ] 룰 기반 대비 개선 확인
- [ ] 다중 분류: cause 8종 또는 grouped 4종 (ego_active/passive/observed/non)

## Phase 7: Scale & Integration
- [ ] ML 모델로 unknown 500건 자동 분류 (스크리닝)
- [ ] 30K 미라벨 영상 coarse-to-fine 적용
- [ ] accident_analysis VLM 재판정 피드백 루프
- [ ] Mask2Former stable class prior + OccAny geometry 융합
- [ ] traffic_rule_checker 연계: 3D 공간 기반 법규 판단
