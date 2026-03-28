# Accident Analysis GT Schema

## 설계 원칙

1. **충돌 단위(per-collision)로 기록** — 연쇄 추돌, 다중 충돌 지원
2. **양쪽 접촉면 명시** — ego/vehicle A의 어디와 vehicle B의 어디가 부딪혔는가
3. **영상에서 관찰 가능한 것만** — 속도, 브레이크, 운전자 의도는 기록하지 않음
4. **OccAny 10개 feature 검증에 필요한 것** — 과도한 텍스트 배제
5. **1건 리뷰 2-3분 이내** — 대규모 라벨링 실현 가능

---

## Contact Zone 체계

차량을 8개 zone으로 나눈다:

```
        front_left   front_center   front_right
            ┌────────────────────────┐
            │                        │
 left_side  │         (roof)         │  right_side
            │                        │
            └────────────────────────┘
        rear_left    rear_center    rear_right
```

| Zone | 설명 |
|------|------|
| `front_center` | 전방 중앙 (범퍼) |
| `front_left` | 전방 좌측 (헤드라이트/펜더) |
| `front_right` | 전방 우측 |
| `left_side` | 좌측면 (도어) |
| `right_side` | 우측면 |
| `rear_center` | 후방 중앙 (범퍼) |
| `rear_left` | 후방 좌측 (테일라이트) |
| `rear_right` | 후방 우측 |
| `unclear` | 영상에서 판단 불가 |

---

## GT Schema (per-video)

```yaml
schema: accident_gt/v1
stem: "20181009_SEQ_S_F_D_1_O_1_0"
reviewer: "ktl"
review_date: "2026-03-28"

# === 영상 수준 ===
involvement: "ego"           # ego / observed / unclear
observability: "clear"       # clear / partial / poor
confidence: "high"           # high / medium / low
n_collisions: 2              # 충돌 횟수

# === 충돌 목록 (per-collision) ===
collisions:
  - collision_id: 1
    time_sec: 3.05                    # 충돌 시점 (초)

    # 충돌 당사자 A
    vehicle_a: "ego"                  # ego / car1 / car2 / truck1 / bus1 / ...
    vehicle_a_type: "car"             # car / truck / bus / motorcycle / bicycle / pedestrian / structure / unknown
    contact_zone_a: "front_center"    # 8-zone

    # 충돌 당사자 B
    vehicle_b: "car1"
    vehicle_b_type: "car"
    contact_zone_b: "rear_center"     # 8-zone

    # 충돌 특성
    collision_type: "rear_end"        # rear_end / side_swipe / head_on / t_bone / single_vehicle / unclear
    severity_visible: "moderate"      # minor (스크래치) / moderate (변형) / severe (대파/전복) / unclear

  - collision_id: 2
    time_sec: 3.8
    vehicle_a: "ego"
    vehicle_a_type: "car"
    contact_zone_a: "front_left"
    vehicle_b: "guardrail"
    vehicle_b_type: "structure"
    contact_zone_b: "unclear"
    collision_type: "single_vehicle"
    severity_visible: "moderate"

# === 사고 원인 (영상 수준) ===
cause: "rear_end_ego_at_fault"        # 8종
camera_shake_visible: true            # 카메라 충격 흔들림 관찰 여부

# === 선택 (추가 분석용) ===
notes: ""                             # 특이사항
```

---

## Collision Type 체계

| Type | 설명 | 예시 |
|------|------|------|
| `rear_end` | 추돌 (같은 방향, 후방 접촉) | ego front → car1 rear |
| `side_swipe` | 측면 접촉 (같은/반대 방향) | ego right_side → car1 left_side |
| `head_on` | 정면 충돌 (반대 방향) | ego front → car1 front |
| `t_bone` | T자 충돌 (직각 방향) | ego front → car1 left_side |
| `single_vehicle` | 단독 사고 (구조물, 전복) | ego front → guardrail |
| `unclear` | 판단 불가 | |

---

## 기존 GT와 비교

### 기존 GT의 문제

| 문제 | 현재 | 개선 |
|------|------|------|
| 단일 충돌만 기록 | collision_time 1개 | **collisions 리스트** (다중 충돌 지원) |
| 접촉면 불명확 | zone이 ego 기준만 (front_left 등) | **양쪽 zone 명시** (A: front_center, B: rear_center) |
| 상대 차량 접촉면 없음 | vehicle="car1" 뿐 | **contact_zone_b** 추가 |
| 충돌 유형 불일치 52% | collision_types vs vlm_zone | **collision_type을 contact zone에서 유도** |
| 텍스트 과다 | description, evidence, contributing_factors | **제거** (Gold에서 불필요) |
| VLM 중복 | vlm_proposal ≈ verified | **verified만 유지** |

### 새 스키마로 만들어낼 수 있는 것

| 분석 목표 | 필요 GT 필드 | 가능? |
|-----------|-------------|-------|
| ego vs observed 분류 | `involvement` | ✅ |
| 충돌 시점 추정 | `collisions[].time_sec` | ✅ (다중 시점) |
| 충돌 방향 분류 | `contact_zone_a` + `contact_zone_b` | ✅ |
| 충돌 유형 분류 | `collision_type` (zone에서 유도 가능) | ✅ |
| 충돌 심각도 | `severity_visible` | ✅ |
| 상대 차량 종류 | `vehicle_b_type` | ✅ |
| 다중 충돌 분석 | `n_collisions`, collisions 리스트 | ✅ |
| OccAny sector 검증 | `contact_zone_a` ↔ sector_collapse | ✅ |
| OccAny rotation 검증 | `collision_type` + zone ↔ rotation_shock 방향 | ✅ |
| 신뢰도 필터링 | `observability`, `confidence` | ✅ |

### 만들어낼 수 없는 것 (GT 범위 밖)

| 분석 목표 | 이유 |
|-----------|------|
| 절대 속도/감속도 | 영상에서 관찰 불가 (physics_sim 추정은 별도) |
| 법적 과실 비율 | 법률 판단, 영상만으로 불가 |
| 부상 심각도 | 차량 내부 상황 관찰 불가 |
| 정확한 충돌 각도 | 3D 재구성 정확도 부족 |

---

## 리뷰 시간 추정

| 영상 유형 | 필드 수 | 예상 시간 |
|-----------|---------|----------|
| 단일 충돌, 명확 | 6 + collision 1개 | 2분 |
| 다중 충돌, 명확 | 6 + collision 2-3개 | 3-4분 |
| 불명확 (poor observability) | 6 + unclear 다수 | 2분 (빠르게 unclear 처리) |

80건 목표 × 평균 3분 = **~4시간** 소요

---

## OccAny Feature ↔ GT 검증 매핑

| OccAny Feature | 검증할 GT 필드 | 검증 방법 |
|----------------|---------------|----------|
| `ego_signal_strength` | `involvement` | ego > observed인가? |
| `sector_collapse` (best_sector) | `contact_zone_a` | 급락 섹터 = 접촉 zone과 일치? |
| `rotation_shock` | `collision_type` | t_bone/side_swipe → 높은 rotation? |
| `left_right_asymmetry` | `contact_zone_a` | left zone → 음수? right → 양수? |
| `approaching_object` | `collision_type=rear_end` | 추돌 시 전방 접근 감지? |
| `flip_rate` | `involvement=ego` + `severity` | ego+severe → 높은 flip? |
| `brightness_change` | `camera_shake_visible` | shake=true → 높은 change? |
| `vehicle_ratio` | `vehicle_b_type` | truck → 높은 ratio? |
| `free_space_ratio` | `collision_type=t_bone` | 교차로 → 높은 free space? |
| `n_collisions` (시계열 급락 횟수) | `n_collisions` | 급락 횟수 = 충돌 횟수? |
