# OccAny Accident Analysis — Architecture & Background

## Overview

OccAny는 **카메라 영상만으로 3D 점유(occupancy) 예측**을 하는 프레임워크(CVPR 2026, Valeo.ai)다.
이 프로젝트에서는 OccAny를 **블랙박스 사고 영상 3D 분석 도구**로 활용한다.

핵심 원칙: OccAny는 **"3D geometric evidence layer"** — 독립적 사고 분석기가 아니라, 기존 VLM/VP 판정을 검증·보강하는 물리적 증거 계층이다 (ADR-001 합의).

---

## 3-Layer Evidence Architecture

사고 영상 분석에 세 개의 독립적 증거 계층이 존재한다.

```
┌─────────────────────────────────────────────────┐
│              Accident Video (dashcam)            │
└──────────┬──────────┬──────────┬────────────────┘
           │          │          │
     ┌─────▼─────┐ ┌──▼───┐ ┌───▼────┐
     │ VP        │ │OccAny│ │ VLM    │
     │(2D frame) │ │(3D)  │ │(text)  │
     └─────┬─────┘ └──┬───┘ └───┬────┘
           │          │          │
     ┌─────▼──────────▼──────────▼────┐
     │      Rule-based Fusion         │
     │   (Phase 3 → future: ML)       │
     └────────────────────────────────┘
```

### Layer 1: VP (VisionPilot) — 2D 영상 특징 추적

VP는 블랙박스 영상에서 **프레임별 2D 특징 변화를 시계열로 추적**한다.

| VP 지표 | 설명 | 사고 감지 정확도 (ego, within_1s) |
|---------|------|--------------------------------|
| `veh_area_change` | 전방 차량 bbox 면적 급변 | **70%** |
| `depth_std_delta` | 깊이 추정값의 분산 급변 | **66%** |
| `combined_risk` | 위험 종합 점수 피크 | **63%** |
| `depth_spike` | 깊이 추정값 급변 | **57%** |
| `nearest_depth_change` | 최근접 물체 깊이 변화 | **43%** |
| `lane_width_change` | 차선 폭 급변 (차선 이탈) | 34% |
| `front_contact` | 전방 접촉 감지 (Mask2Former) | 50% |

**Risk Events**: VP는 프레임별 위험 이벤트(`close_approach`, `depth_drop` 등)를 시간대별로 기록한다.

**특성**:
- 2D 이미지 좌표 기반 (깊이는 상대적 추정)
- 프레임 단위 시계열 분석 (10-30fps)
- 차선 검출 기반 차선 이탈 감지

### Layer 2: OccAny — 3D 공간 복원

OccAny는 연속 프레임에서 **밀집 3D 포인트 클라우드와 카메라 포즈를 복원**한다.

| OccAny 지표 | 설명 | 단위 |
|-------------|------|------|
| `ego_signal_strength` | 3D 복원 품질 급락 (전체) | 0~1 |
| `sector_collapse` | 섹터별 밀도 급락 (5개 섹터) | 0~1 |
| `rotation_shock` | 카메라 회전 급변 | radians |
| `left_right_asymmetry` | 좌우 밀도 급락 비대칭 | -1~+1 |
| `approaching_object` | 전방 깊이 연속 감소 추세 | boolean |
| `free_space_ratio` | 중앙 시야 빈 공간 비율 | 0~1 |
| `reliability` | 복원 신뢰도 (품질 게이팅) | 0~1 |
| `min_depth` | 전방 최소 거리 (상대적) | 상대 m |

**특성**:
- 3D 월드 좌표 기반 (깊이는 밀집 포인트 클라우드에서 직접)
- 5프레임 윈도우 단위 분석 (Must3R O(n²) 메모리 제약)
- 캘리브레이션/GPS 없이 동작 — **scale-free 상대 특징만 신뢰**

### Layer 3: VLM — 텍스트 기반 장면 이해

VLM은 영상 프레임을 보고 **자연어로 장면을 설명**한다.

| VLM 출력 | 설명 |
|----------|------|
| `accident/non_accident` | 사고 발생 여부 (F1=0.757) |
| `evidence` | "전방 차량에 급접근, 파편 발생" 등 텍스트 |
| `collision_time` | 추정 사고 시점 (초) |
| `confidence` | 판정 확신도 |

**특성**:
- 2D 시각 + 상식 기반 추론
- 정성적 판단 (정량적 수치 아님)
- Open-ended (새로운 상황에 유연)

---

## 세 Layer의 관계

| 비교 축 | VP | OccAny | VLM |
|---------|-----|--------|-----|
| **차원** | 2D (이미지) | **3D (월드)** | 텍스트 |
| **깊이 정보** | 상대적 추정 | **밀집 포인트** | 정성적 |
| **방향 분석** | 제한적 | **섹터별 비대칭** | "왼쪽에서 접근" |
| **회전 감지** | 없음 | **rotation shock** | "카메라 흔들림" |
| **시간 해상도** | 프레임 단위 | 5프레임 윈도우 | 수초 단위 |
| **시맨틱** | Mask2Former (안정) | SAM2 (불안정) | 자연어 |
| **캘리브레이션** | 불필요 | 불필요 | 불필요 |
| **상호 상관** | — | VP-OccAny: **r=0.045 (독립)** | — |

**VP와 OccAny가 거의 독립적** (r=0.045) → 앙상블 시 상호 보완 가치가 높다.

---

## 사고 원인 분류 (cause_classification)

157건의 사고에 대해 8가지 원인이 라벨링되어 있다.

| 사고 원인 | 건수 | 주체 | ego_signal |
|-----------|------|------|------------|
| `rear_end_ego_at_fault` | 56 | ego (능동) | 0.180 |
| `observed_accident` | 58 | observed | 0.141 |
| `unsafe_lane_change_ego` | 14 | ego (능동) | 0.224 |
| `cut_in_other` | 11 | ego (피동) | 0.135 |
| `solo_collision` | 6 | ego (능동) | 0.184 |
| `sudden_stop_front` | 5 | ego (피동) | 0.163 |
| `rear_end_other_at_fault` | 4 | ego (피동) | 0.057 |
| `side_collision_intersection` | 3 | ego (능동) | 0.293 |

### 4-Group 분류

| 그룹 | 정의 | n | ego_signal mean |
|------|------|---|-----------------|
| **ego_active** | ego가 능동적으로 충돌 (추돌, 차선변경, 단독) | 56 | **0.196** |
| **ego_passive** | ego가 피동적 (상대 과실, 끼어들기) | 15 | 0.127 |
| **observed** | 타 차량 사고 관찰 | 50 | 0.141 |
| **non_accident** | 사고 아님 (대조군) | 26 | 0.151 |

통계적 유의성:
- ego_active vs ego_passive: **p=0.043***
- ego_active vs observed: **p=0.047***
- ego_active vs non_accident: p=0.155 (유의하지 않음)

---

## 판별 전략: 룰 기반 → ML 전환

### 현재: 룰 기반 (Phase 3)

```
ego_vs_observed:
  IF max_sector_collapse > threshold AND reliability > 0.5 → "likely_ego"

impact_side:
  IF left_collapse > right_collapse * 1.5 → "left"
  IF right_collapse > left_collapse * 1.5 → "right"
  ELSE → "front" or "unclear"

approaching_object:
  IF front_center_depth 연속 3프레임 감소 → True

collision_type_hint:
  IF rotation_shock > 30° AND asymmetry > 0.3 → "side_collision"
  IF rotation_shock < 5° AND front_center_collapse > 0.2 → "rear_end"
```

### 룰 기반인 이유

1. **데이터 부족**: 147건은 ML 학습에 불충분 (overfitting 위험)
2. **디버깅 가능**: "왜 이 건을 틀렸는가" 즉시 추적 가능
3. **GT 품질 문제**: 52% 충돌유형 불일치 — 노이즈 데이터로 ML 학습은 위험
4. **독립성 활용**: VP(r=0.045)와 독립적이므로 각각 룰 기반 후 앙상블이 효과적

### ML 전환 조건 (향후)

- Gold GT 60-70건 확보 (수동 검증)
- 룰 기반 baseline F1 확인
- VP features + OccAny features 결합 시 개선 확인
- → logistic regression 또는 gradient boosting

---

## 데이터 현황

### 전체 데이터

| 소스 | 건수 | 설명 |
|------|------|------|
| 전체 영상 | 30,073 | /data2/accident_data/ |
| review 완료 | 1,099 | accident/non/unknown/inadequate |
| accident 확정 | 158 | collision 분석 대상 |
| non_accident 확정 | 30 | 대조군 |
| unknown | 500 | 판정 불가 → 스크리닝 후보 |
| inadequate | 410 | 제외 (영상 품질 부적합) |

### OccAny 처리 완료

| 데이터셋 | 건수 | 결과 |
|----------|------|------|
| Phase 2 (accident, daytime) | 121 | ego_signal F1=0.642 |
| Phase 2 (non_accident, daytime) | 26 | 대조군 signal=0.151 |
| Phase 3 (dense windows, 진행중) | 121 | sectorized features |

### 라벨 파일 전체 인벤토리 (accident_analysis project)

#### 핵심 라벨 (분류/원인)

| 파일 | 건수 | 내용 | 생성 |
|------|------|------|------|
| `review_results.json` | 1,099 | 사고/비사고/unknown/inadequate 분류 | VLM 자동 |
| `collision_analysis_results.json` | 157 | ego/observed, impact zone, description | VLM+휴리스틱 |
| `cause_classification_results.json` | 157 | 사고 원인 8가지, evidence, contributing_factors | VLM 자동 |
| `collision_types.json` | 125 | 충돌 유형 (rear/side/front/unclear) | VLM 자동 |

#### Perception 기반 검증

| 파일 | 건수 | 내용 | 생성 |
|------|------|------|------|
| `mask_contact_eval_results.json` | 157 | **mask_zone** (Mask2Former 기반 충돌 zone), mask_vehicle, mask_impact_time | Panoptic 자동 |
| `risk_event_results.json` | 158 | VP 프레임별 위험 이벤트 (close_approach, depth_drop) | VP 자동 |
| `physics_sim_results.json` | 100 | **속도 추정** (ego/other/impact speed, TTC range), scenario template | 시뮬레이션 |
| `feature_kpi_results.json` | — | VP 정량 KPI (depth, lane, risk score) | VP 자동 |

#### 수동 검증 (Gold GT 후보)

| 파일 | 건수 | 내용 |
|------|------|------|
| `gt_alignment_results.json` | 15 | 수동 GT + VP/panoptic 교차 검증 (collision_type, reasons, zone match) |
| `expert_review_20.json` | 20 | 전문가 리뷰 대상 영상 목록 |
| `gt_review_checklist.json` | 18 | GT 재검수 대상 (ADR-011) |

#### VLM 평가

| 파일 | 건수 | 내용 |
|------|------|------|
| `vlm_accident_results.json` | 188 | VLM 사고 판정 + confidence |
| `vlm_assessment_results.json` | 138 | VLM 종합 평가 |
| `vlm_rerank_results.json` | 60 | VLM 재순위 평가 |
| `collision_type_vlm_eval.json` | 112 | 충돌 유형 GT vs VLM 비교 |
| `report_evals.json` | 85 | 보고서 품질 평가 |

#### 주요 발견: mask_contact_eval

**mask_zone**(Mask2Former 기반)은 VLM zone과 **16%만 일치**:

| mask_zone (Panoptic) | 건수 | vlm_zone과 비교 |
|---------------------|------|----------------|
| front_center | 55 | VLM은 front_center를 28건만 |
| front_left | 31 | |
| front_right | 28 | |
| left_side | 13 | |
| right_side | 13 | |
| None (미감지) | 17 | |

→ VLM과 Panoptic의 zone 판정이 크게 다름. **OccAny의 sector_collapse가 제3의 독립적 zone 판정** 역할 가능.

#### 주요 발견: physics_sim_results

100건에 대해 **물리 시뮬레이션 기반 속도/TTC 추정**:
- ego_speed_range_kmh: [32-78] km/h
- impact_speed_range_kmh: [0-98] km/h
- ttc_range_sec: [0.8-3.5] s

→ OccAny의 상대적 접근 속도와 교차 검증 가능 (절대값은 불가하지만 순서/비율 비교).

### GT 4단계 상세도

| Level | 파일 | 건수 | 필드 | 신뢰도 |
|-------|------|------|------|--------|
| **L1** | review_results | 1,099 | label만 | VLM 자동 (낮음) |
| **L2** | collision_analysis | 157 | + time, subject, impacts, description | VLM+휴리스틱 (중간) |
| **L3** | cause + mask_contact + physics_sim | 157/100 | + cause, zone, speed, TTC | 자동+시뮬 (중간) |
| **L4** | gt_alignment | **15** | + 수동 GT + 교차 검증 | **사람 검증 (높음)** |

### Gold GT 필요 필드 (ML용)

```yaml
stem: "20181009_SEQ_S_F_D_1_O_1_0"
reviewer: "human"
confidence: "high"        # high / medium / low
observability: "clear"    # clear / partial / poor

# 필수 (OccAny 10 feature 검증 대상)
accident_subject: "ego"   # ego / observed
collision_time: 3.05      # seconds
impact_side: "front"      # front / left / right / rear / unclear
cause: "rear_end_ego_at_fault"  # 8종 중 1

# 선택 (추가 검증용)
counterpart_type: "car"           # car / truck / VRU / structure / unknown
counterpart_sector: "front_center"  # last visible position
camera_shake_visible: true
post_impact_blocked: false
```

### Gold GT 구축 전략

1. **기존 L4 (15건) + expert_review (20건) + gt_checklist (18건)** = 시작점 ~35건
2. **추가 45건 선정**: VLM/mask/OccAny 불일치 건 우선
3. **2-stage review**: blind → assisted (Phase 3 evidence 활용)
4. **목표**: 80건 Gold, 카테고리당 최소 10건
5. **ML 학습**: Gold 80 + Silver(L2/L3) ~150 = 230건으로 간단한 모델 학습

---

## Semantic Segmentation 정보

### 클래스 체계 (nuScenes 17+1 classes)

| ID | Class | Category |
|----|-------|----------|
| 0 | unknown | unknown |
| 1 | fence | structure |
| 2 | bicycle | VRU |
| 3-5, 9 | other-vehicle | vehicle |
| 4 | car | vehicle |
| 6 | motorcycle | vehicle |
| 7 | person | VRU |
| 8 | traffic-sign | structure |
| 10 | truck | vehicle |
| 11 | road | road_surface |
| 12 | other-ground | road_surface |
| 13 | sidewalk | road_surface |
| 14 | terrain | nature |
| 15 | building | structure |
| 16 | vegetation | nature |
| 17 | empty | empty |

### 사고 원인별 시맨틱 구성

| 원인 | vehicle | road | nature | empty | flip rate |
|------|---------|------|--------|-------|-----------|
| rear_end_ego_at_fault | 24.6% | 27.3% | 9.0% | 36.6% | 19.4% |
| unsafe_lane_change_ego | 23.6% | 31.0% | 6.5% | 36.5% | **28.2%** |
| solo_collision | 31.9% | 23.0% | 7.5% | 35.6% | **32.9%** |
| observed_accident | 25.6% | 24.3% | 10.4% | 37.6% | 17.9% |
| rear_end_other_at_fault | **43.8%** | 16.7% | 3.2% | 36.1% | **10.5%** |
| side_collision_intersection | 12.3% | **33.2%** | 9.6% | **44.8%** | 19.7% |

### 시맨틱 패턴 발견

1. **rear_end_other_at_fault**: vehicle 비율 최고(43.8%), flip rate 최저(10.5%)
   - 해석: 전방 차량이 가까이 있고 안정적 → 정지 중 뒤에서 추돌당한 상황
2. **solo_collision / unsafe_lane_change**: flip rate 최고(28-33%)
   - 해석: 카메라가 급격히 움직이면서 시맨틱 불안정 → 능동적 사고의 시그널
3. **side_collision_intersection**: vehicle 비율 최저(12.3%), empty 최고(44.8%)
   - 해석: 교차로에서 열린 공간이 많고 차량은 측면에서 접근
4. **observed_accident vs ego**: vehicle 비율 비슷(25-26%), flip rate로 구분 가능(18% vs 19-33%)

### Semantic Flip Rate — 시맨틱 불안정도

- **전체 평균 20.3%**: 프레임간 non-empty 픽셀의 20%가 클래스 변경
- 이는 SAM2의 프레임별 독립 추론 때문 (video mode에서도 발생)
- **flip rate 자체가 사고 특성을 반영하는 feature**:
  - 높은 flip rate = 카메라 급변/충격 = ego 능동 사고 경향
  - 낮은 flip rate = 안정적 시야 = 피동/관찰 사고 경향
- 픽셀 단위 시맨틱 추적은 불가 → **카테고리별 비율과 flip rate만 feature로 활용**

### 활용 가능한 시맨틱 Feature

| Feature | 설명 | 사고 분석 활용 |
|---------|------|--------------|
| `vehicle_ratio` | 차량 클래스 비율 | 높으면 근접 차량 존재 (추돌 시그널) |
| `road_ratio` | 도로면 비율 | 높으면 개활지 (교차로/고속도로) |
| `empty_ratio` | 빈 공간 비율 | 높으면 3D 복원 실패 or 하늘 |
| `flip_rate` | 프레임간 클래스 변경률 | **능동 사고 시그널** (ego_signal 보완) |
| `vehicle_ratio_change` | 시간에 따른 vehicle 비율 변화 | 접근/이탈 감지 |
| `VRU_present` | 보행자/자전거 존재 여부 | VRU 관련 사고 감지 |

---

## OccAny 출력 데이터 전체 인벤토리

### pts3d_render.npy

| Key | Shape | 활용 | Feature |
|-----|-------|------|---------|
| `pts3d` | (N, H, W, 3) | ✅ | min_depth, spatial_extent, sector depth |
| `conf` | (N, H, W) | ✅ | ego_signal, density, sector collapse, reliability |
| `c2w` | (N, 4, 4) | ✅ | rotation shock, velocity, trajectory |
| `semantic_2ds` | (N, H, W) | ✅ | class ratios, flip rate |
| `colors` | (N, H, W, 3) | ✅ | **brightness change** (신규) |
| `focal` | (N,) | ✗ | 프레임간 동일 (상수) — 변별력 없음 |
| `pts3d_local` | (N, H, W, 3) | ✗ | pts3d와 거의 동일 (diff=0.12) — 중복 |

### voxel_predictions.pkl

| Key | Shape | 활용 | 비고 |
|-----|-------|------|------|
| `estimated_input_camera_poses` | (N, 4, 4) | ✅ | c2w와 동일 |
| `estimated_input_intrinsics` | (N, 3, 3) | ✗ | 프레임간 동일 — 변별력 없음 |
| `estimated_input_images` | (N, H, W, 3) | ✅ | 시각화, review 도구 |
| `voxel_size`, `voxel_origin` | scalar, (3,) | ✗ | 설정값 |
| `render_th2.0` | (200, 200, 24) | ✗ | 100% occupied — threshold 문제로 변별력 없음 |

### 활용 가능한 Feature 전체 목록 (10개)

| # | Feature | Source | 사고 패턴 | 통계적 근거 |
|---|---------|--------|----------|------------|
| 1 | `ego_signal_strength` | conf | ego/observed 구분 | p=0.017, F1=0.642 |
| 2 | `sector_collapse` (5개) | conf × sector | 충돌 방향 추정 | Phase 3 테스트 확인 |
| 3 | `rotation_shock` | c2w | 충격 강도/방향 | solo=115°, rear_end_other=1.3° |
| 4 | `left_right_asymmetry` | sector collapse | 측면 충돌 방향 | solo=-0.58 (우측) |
| 5 | `approaching_object` | pts3d depth 추세 | 전방 접근 감지 | boolean |
| 6 | `reliability` | conf, rotation, free_space | 결과 품질 게이팅 | 0~1 |
| 7 | `vehicle_ratio` | semantic | 근접 차량 존재 | rear_end_other=43.8% (최고) |
| 8 | `flip_rate` | semantic 프레임간 변동 | 능동 사고 시그널 | solo=33%, other=10.5% |
| 9 | `brightness_change` | colors | 능동 사고 시그널 | solo=0.118, other=0.019 |
| 10 | `free_space_ratio` | conf (중앙 시야) | 시야 차단 | intersection=44.8% |

### Feature 간 상관 패턴

flip_rate와 brightness_change는 같은 방향의 패턴을 보인다:
- **높음**: solo_collision (카메라 급변) > unsafe_lane_change (급기동) > rear_end_ego (충격)
- **낮음**: observed (안정적 관찰) > rear_end_other (정지 중 피추돌)

이는 두 feature가 모두 **"카메라의 물리적 동요"**를 다른 각도에서 측정하기 때문.
ego_signal(conf 급락), flip_rate(시맨틱 변동), brightness_change(색상 변동)는 상관이 있을 수 있으나, 각각 다른 원인으로도 발생하므로 앙상블 시 보완적.

---

## OccAny의 능력 범위

### 신뢰할 수 있는 것 (scale-free 상대 특징)
- 좌/우/전방 비대칭 (섹터별 밀도 급락)
- 가까움/먼 순서 (상대적 깊이)
- 주행 공간 폐쇄 (free-space collapse)
- 카메라 회전 충격 (rotation shock)
- 접근 방향의 부호 (접근/이탈)
- 사고 후 시야 차단 (post-impact blockage)

### 주의가 필요한 것 (캘리브레이션 없음)
- 절대 거리 (미터)
- 절대 속도 (m/s)
- TTC 절대값 (초)
- 씬 간 이동 거리

### 불가능한 것
- 정확한 충돌 각도, 법적 과실
- 브레이크 힘, delta-v, 부상 심각도
- 카메라 밖 사건
- 운전자 의도, 교통법규 준수 여부
- 정밀 물체 궤적 (트래커 없이)

---

## 프로젝트 간 관계

```
accident_analysis (라벨, VLM, 원인 분류)
       │
       ├── VP inference (2D 특징, risk events)
       │
       └── OccAny (3D 증거) ← 이 프로젝트
              │
              ├── perception (Mask2Former → 안정적 시맨틱)
              ├── vehicle-tracker (YOLO+BYTETrack → 물체 추적)
              └── traffic_rule_checker (법규 판단)
```
