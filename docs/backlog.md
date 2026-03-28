# Backlog

## improvement

- [ ] VLM + OccAny 3D 연계 분석 (발견일: 2026-03-28)
  - accident_analysis의 VLM 판정 (2D 텍스트 기반)에 OccAny 3D 메트릭을 결합
  - VLM evidence ("전방 차량에 접근") + OccAny min_depth 시계열 = 정량적 근거 보강
  - VLM이 ego/observed 불확실할 때 ego_signal_strength로 tie-breaking
  - VLM confidence가 낮은 케이스에 3D 물리 지표를 추가 feature로 사용

- [ ] SAM2 시맨틱 시간 일관성 문제 해결 (발견일: 2026-03-28)
  - 현상: OccAny의 SAM2가 프레임마다 같은 객체에 다른 class ID를 부여
  - 원인: SAM2는 프레임 독립 추론, temporal consistency 보장 없음
  - 해결 방안:
    1. OccAny의 3D 포인트 매칭으로 프레임 간 같은 3D 위치의 클래스를 majority voting
    2. perception 프로젝트의 Mask2Former temporal smoothing (ADR-010) 적용
    3. video propagation 기반 tracking (SAM2의 video mode 활용)
  - 기대 효과: 시맨틱 BEV 맵 품질 향상, 사고 장면의 객체별 궤적 추적 가능

- [ ] Mask2Former vs SAM2 시맨틱 비교 (발견일: 2026-03-28)
  - perception 프로젝트의 Mask2Former panoptic segmentation과 OccAny SAM2 결과 비교
  - Mask2Former: 고정된 클래스셋, temporal smoothing 적용 가능
  - SAM2: open-vocabulary, 프레임별 독립 추론
  - 상호 보완: Mask2Former의 안정적 클래스 + SAM2의 유연한 탐지
  - 관련: perception Phase 3 (Semantic Segmentation)

- [ ] 사고 영상 전체 157건 배치 처리 (발견일: 2026-03-28)
  - 현재 35건 (warning+daytime) 처리 중
  - 나머지 122건 (야간 포함, warning 없는 건)도 처리하여 통계적 유의성 확보
  - 예상 소요: ~3시간 (A100 1대)

- [ ] 다중 지표 판별 모델 (발견일: 2026-03-28)
  - ego_signal_strength 단일 지표 대신 다중 지표 결합
  - features: ego_signal, min_depth, ttc, velocity, semantic_change_rate
  - 간단한 logistic regression 또는 decision tree로 ego/observed 분류
  - 157건 완료 후 train/test split으로 평가

## idea

- [ ] 사고 영상 3D 복원 → 사고 재구성 시뮬레이션 (발견일: 2026-03-28)
  - OccAny 포인트 클라우드 + 카메라 궤적으로 사고 순간의 3D 장면 재구성
  - 시간별 3D 스냅샷을 연결하면 "사고 재현 애니메이션" 생성 가능
  - traffic_rule_checker와 연계: 3D 공간에서 법규 위반 여부 판단

- [ ] 3DGS 초기화에 OccAny 포인트 활용 (발견일: 2026-03-28)
  - drivestudio의 3D Gaussian Splatting 초기점으로 OccAny dense 포인트 사용
  - COLMAP SfM 대비 빠르고 안정적인 초기화 가능성
