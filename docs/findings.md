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


## 2026-03-28: OccAny 입력 제약 사항 (RTB 데이터)

**맥락**: RTB recording camera_front 595프레임으로 OccAny inference 테스트

**발견**:
- Must3R는 O(n²) 메모리로 pairwise 매칭 → **5프레임 이상은 A100 80GB에서 OOM**
- 5프레임 단위 씬 분할 + 순차 처리로 해결 (119씬, 7분)
- third_party/croco/models/__init__.py 누락 → 수동 생성 필요
- `--silent` 모드에서 체크포인트 경로가 `occany.pth` (symlink으로 해결)
- 4112x2176 → 512x272로 자동 리사이즈 (고해상도 장점 상실)

**영향**: 대량 처리 시 5프레임 단위 분할 필수. 배치 파이프라인에 반영 완료.
