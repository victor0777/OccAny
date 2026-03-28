# Findings

## 2026-03-28: OccAny 사고 영상 3D 분석 — ego vs observed 판별 가능성

**맥락**: accident_analysis 라벨링된 157건 사고 영상에 OccAny (Must3R + SAM2) inference를 적용하여 3D 지표를 추출하고, ego/observed 사고 판별에 활용 가능한지 검증

**발견**:

### OccAny 추출 지표

| 지표 | 설명 | 단위 |
|------|------|------|
| `min_depth` | 전방 FOV 중앙 1/3의 5th percentile 깊이 | m |
| `density` | conf > 3.0인 픽셀 비율 (3D 복원 품질) | 0~1 |
| `cam_pos` | 모델 추정 카메라 3D 위치 | m |
| `spatial_extent` | 포인트 클라우드 X/Y/Z 범위 | m |
| `semantic_ratios` | SAM2 시맨틱 클래스별 픽셀 비율 | 0~1 |

### 파생 판별 메트릭

| 메트릭 | 계산 | 활용 |
|--------|------|------|
| `ego_signal_strength` | 연속 씬 간 density 최대 급락폭 | ego/observed 판별 핵심 |
| `ttc_estimate` | min_depth / 접근속도 (depth 감소 구간) | 충돌까지 남은 시간 |
| `velocity` | 연속 cam_pos 차이 / dt | 속도 추정 |

### 최종 결과 (35건: ego=23, observed=12, warning+daytime)

| | ego (n=23) | observed (n=12) |
|---|---|---|
| 평균 signal | **0.178** | 0.072 |
| 중앙값 | **0.139** | 0.059 |
| 표준편차 | 0.157 | 0.055 |
| 범위 | [0.000, 0.568] | [0.000, 0.167] |
| **t-value** | **2.92 (통계적 유의)** | |

### 최적 Threshold

| threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.08 (최적) | **81%** | **74%** | **0.773** |
| 0.15 | 85% | 48% | 0.611 |
| 0.18 | 100% | 35% | 0.516 |

### 충돌 유형별 signal

| 유형 | n | mean | median |
|------|---|------|--------|
| unknown | 6 | 0.187 | 0.159 |
| rear_collision | 8 | 0.161 | 0.130 |
| front_collision | 5 | 0.153 | 0.118 |
| side_collision | 12 | 0.125 | 0.118 |
| observed_accident | 3 | 0.065 | 0.077 |

### 한계

- ego인데 signal=0인 케이스 5건: 3D 복원 자체 실패 (min_depth=N/A). 정지 상태 충돌 또는 영상 품질 문제
- observed인데 signal 높은 케이스 (회피 기동 중 카메라 흔들림)
- 단일 지표(density 급락)만으로는 F1=0.773 한계 → 다중 지표 결합 필요

**영향**: ego/observed 불확실 45건에 대한 3D 기반 재판정 파이프라인 구축 가능. 기존 VLM 2D 판정에 3D 물리 증거를 보강하는 역할. threshold=0.08에서 Precision 81%, F1=0.773 달성.


## 2026-03-28: ego_signal_strength의 한계 — 사고/비사고 분류 불가

**맥락**: non_accident 26건 (GT 확정)을 대조군으로 OccAny 추론하여 ego_signal_strength 비교

**발견**:

### 3-way 비교 (ego=36, observed=25, non_accident=26)

| 그룹 | mean | median | std |
|------|------|--------|-----|
| ego accident | **0.212** | 0.173 | 0.170 |
| non_accident | 0.151 | 0.140 | 0.118 |
| observed | 0.101 | 0.062 | 0.103 |

- ego vs observed: t=3.12, **p=0.003 (유의)**
- ego vs non_accident: t=1.64, **p=0.105 (유의하지 않음)**
- non_accident 26건 중 12건(46%)이 signal > 0.15

### 원인

ego_signal_strength는 **"카메라 충격/흔들림"을 감지**하는 것이지 "사고"를 감지하는 것이 아님.
non_accident에서도 급정거, 과속방지턱, 비포장 도로, 급차선 변경 등으로 3D 복원 품질 급락이 빈번.

### 결론

| 용도 | ego_signal 유효? |
|------|----------------|
| 사고 내 ego vs observed 구분 | **유효** (p=0.003) |
| accident vs non_accident 분류 | **불충분** (p=0.105) |

→ ego_signal은 **"사고가 확인된 건"에서 ego/observed 판별**에만 사용.
→ 사고/비사고 분류에는 추가 특징 필요 (시맨틱 변화, 장면 구조 파괴, 파편 감지 등)

**영향**: Phase 3에서 ego_signal을 사고 감지가 아닌 **사고 내 역할 분류**에 한정해서 사용. 사고/비사고 스크리닝에는 VLM 결과를 1차 필터로 유지.


## 2026-03-28: 121건 최종 결과 — ego_signal 단독 한계 확인

**맥락**: 주간 사고 121건 (ego=62, observed=59) + non_accident 26건 전체 처리 완료

**발견**:

### 최종 통계

| 그룹 | n | mean | median | std | range |
|------|---|------|--------|-----|-------|
| ego accident | 62 | **0.194** | 0.164 | 0.156 | [0.000, 0.568] |
| observed | 59 | 0.134 | 0.114 | 0.113 | [0.000, 0.444] |
| non_accident | 26 | 0.151 | 0.140 | 0.118 | [0.000, 0.483] |

- ego vs observed: t=2.43, **p=0.017 (유의)**
- ego vs non_accident: t=1.40, p=0.165 (유의하지 않음)
- Best F1 (ego vs observed): **0.642** at threshold=0.03

### 35건 → 121건 스케일업에서 성능 저하

| | 35건 (warning only) | 121건 (전체 주간) |
|---|---|---|
| ego mean | 0.178 | 0.194 |
| observed mean | 0.072 | **0.134** (↑86%) |
| p-value | 0.003 | 0.017 |
| Best F1 | 0.773 | **0.642** (↓17%) |

원인: 초기 35건은 "accept_with_warning" = 판별이 애매한 케이스만 → 상대적으로 쉬운 집합이었음. 전체로 확장하면 observed에서도 signal이 높은 케이스 다수 (급정거 동반 관찰 사고 등).

### 충돌 유형별: 차이 미미

rear(0.184), side(0.176), front(0.167) 모두 비슷. 유일하게 **observed_accident(0.061)만 확실히 분리**.

### 결론

1. ego_signal 단독 F1=0.642 → **실용 수준 미달, 추가 특징 결합 필수**
2. Phase 3 sectorized features + reliability gating의 필요성 실증
3. ego_signal은 **다중 특징 앙상블의 한 입력**으로 활용
4. observed_accident 하위 유형만은 ego_signal 단독으로 분리 가능

**영향**: Phase 3 fast track (sectorized features, reliability, VLM verifier) 착수 근거 확보. 단일 지표가 아닌 다중 지표 접근이 필수.


## 2026-03-29: Phase 3 Dense Analysis 121건 — 다중 feature 결합 한계

**맥락**: collision_time ± 5s dense windows (stride 1) + sectorized features 121건 분석 완료

**발견**:

### Feature별 통계적 유의성

| Feature | ego_active mean | observed mean | p-value | 유의? |
|---------|----------------|---------------|---------|-------|
| rotation_shock (deg) | **21.7** | 9.4 | **0.033** | * |
| sector_collapse | 0.110 | 0.082 | 0.089 | ns |
| abs_asymmetry | 0.272 | 0.254 | 0.706 | ns |
| approaching | 82% | 75% | 0.367 | ns |
| reliability | 0.469 | 0.521 | 0.402 | ns |

→ **rotation_shock만 유의미** (p<0.05). 나머지 sectorized features는 유의하지 않음.

### 다중 feature 결합 효과

| 방법 | Best F1 |
|------|---------|
| ego_signal 단독 (Phase 2) | 0.632 |
| sector_collapse 단독 (Phase 3) | 0.634 |
| rotation_shock 단독 (Phase 3) | **0.646** |
| 3개 weighted 결합 | **0.646** |

→ **결합해도 개선 없음.** rotation_shock가 이미 최선이고, 나머지를 더해도 노이즈만 추가.

### 결론

1. OccAny 단독 ego_active 분류: **F1≈0.65가 현실적 상한**
2. **rotation_shock가 가장 유의미한 새 feature** — ego_signal과 독립적 정보 제공
3. sector_collapse는 기대보다 변별력 낮음 — dense window에서도 분포 겹침 큼
4. best_collapse_sector → impact_side 예측 실패 — 균일 분포
5. **VP features와 결합 시 개선 가능** (독립적 r=0.045) — accident_analysis 프로젝트에서 진행

**영향**: OccAny 프로젝트 자체에서의 feature 탐색은 여기서 수렴. 향후 개선은 (1) VP/VLM과의 cross-project 결합, (2) Gold GT 확보 후 ML 전환, (3) 차량 추적/차선 인식 추가에 의존.


## 2026-03-28: dgx01 추가 데이터 확보 + 사고 원인별 3D 패턴

**맥락**: dgx01 accident_analysis에서 추가 라벨 파일 6종 확보 (cause_classification, risk_events, gt_alignment 등)

**발견**:

### 사고 원인 8가지 분류 × ego_signal

| 사고 원인 | n | signal mean | signal median |
|-----------|---|-------------|---------------|
| side_collision_intersection | 3 | **0.293** | 0.304 |
| unsafe_lane_change_ego | 13 | **0.224** | 0.229 |
| solo_collision | 5 | 0.184 | 0.129 |
| rear_end_ego_at_fault | 35 | 0.180 | 0.127 |
| sudden_stop_front | 4 | 0.163 | 0.166 |
| observed_accident | 50 | 0.141 | 0.108 |
| cut_in_other | 8 | 0.135 | 0.178 |
| rear_end_other_at_fault | 3 | **0.057** | 0.056 |

### 패턴

1. **ego 능동 사고 > ego 피동 사고**: ego가 직접 움직인 사고(lane_change 0.224, rear_end_ego 0.180)가 상대 과실(rear_end_other 0.057, cut_in 0.135)보다 signal 높음
2. **교차로 측면 충돌이 가장 높음** (0.293) — 강한 물리적 충격
3. **rear_end_other_at_fault가 가장 낮음** (0.057) — ego는 정지/서행 중 뒤에서 추돌당함 → 카메라 충격 약함
4. **observed_accident** (0.141)이 non_accident (0.151)과 비슷 — ego_signal로는 관찰 사고 ≈ 비사고

### 추가 데이터

- `gt_alignment_results.json` (15건): 수동 검증 GT (collision_type, accident_reasons 포함) → Gold subset 핵심 후보
- `risk_event_results.json` (158건): 프레임별 위험 이벤트 (close_approach, depth_drop 등) → Phase 3 VLM verifier의 비교 기준
- `cause_classification_results.json` (157건): 사고 원인 + contributing_factors → 다중 지표 판별 모델의 target label

**영향**: cause_classification의 8가지 원인 분류가 OccAny 분석의 더 적합한 평가 축. ego/observed 이분법보다 세분화된 원인별 3D 패턴이 실용적. gt_alignment 15건은 Gold GT sprint의 시작점.


## 2026-03-28: OccAny 입력 제약 사항 (RTB 데이터)

**맥락**: RTB recording camera_front 595프레임으로 OccAny inference 테스트

**발견**:
- Must3R는 O(n²) 메모리로 pairwise 매칭 → **5프레임 이상은 A100 80GB에서 OOM**
- 5프레임 단위 씬 분할 + 순차 처리로 해결 (119씬, 7분)
- third_party/croco/models/__init__.py 누락 → 수동 생성 필요
- `--silent` 모드에서 체크포인트 경로가 `occany.pth` (symlink으로 해결)
- 4112x2176 → 512x272로 자동 리사이즈 (고해상도 장점 상실)

**영향**: 대량 처리 시 5프레임 단위 분할 필수. 배치 파이프라인에 반영 완료.
