# Roadmap

## Phase 1: 환경 구축 + 데이터 호환성 검증
- [x] OccAny inference 환경 구축 (체크포인트, 의존성, 패치)
- [x] RTB camera_front 데이터 추론 성공 (5프레임, 595프레임)
- [x] 사고 영상 (야간/주간) 추론 성공
- [x] 3D 분석 지표 추출 파이프라인 프로토타입

## Phase 2: accident_analysis 대규모 분석 ← **current**
- [x] 배치 추론 스크립트 구축 (batch_accident_inference.py)
- [ ] warning 35건 (주간) 추론 완료 + ego/observed 판별 threshold 확정
- [ ] 전체 157건 추론 + 통계적 유의성 확보
- [ ] ego/observed 재판정 결과 → accident_analysis에 피드백

## Phase 3: VLM + 3D 연계 분석
- [ ] VLM evidence에 3D 정량 메트릭 보강
- [ ] 충돌 유형 불일치 58건 3D 기반 재분류
- [ ] 다중 지표 판별 모델 (ego_signal + depth + ttc + velocity)

## Phase 4: 시맨틱 일관성 개선
- [ ] SAM2 프레임 간 class 불일치 해결 (3D majority voting)
- [ ] Mask2Former vs SAM2 비교 실험
- [ ] 시맨틱 BEV 맵 생성 파이프라인
