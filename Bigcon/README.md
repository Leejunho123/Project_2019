## 빅콘테스트 Analysis 퓨처스리그
- Contents
    + 빅콘테스트 소개자료 : 빅콘테스트 퓨처스리그 설명자료.pptx
    + 목표 : 공항별, 항공기별 데이터를 이용하여 항공기 지연 예측
    + 기간 : 2018년 7월 ~ 2018년 9월 (2개월)
    + 팀 인원 : 5
    + 출제 목적
        1. 여객 대기시간 감소 및 공항 혼잡 방지
        2. 공항 자원 운영의 효율화 추진
        3. 효율적 시간배분으로 운항편수 추가 확보 도모
        4. 항공기 대기시간 등으로 추가 발생하는 연료 절감
        5. 환경 영향 감축
    + 사용한 모델
        1. Logistic regression
        2. Xgboost
        3. LightGBM
        + xgboost 와 LightGBM 의 AUROC값이 비슷했으나 LightGBM의 속도가 압도적으로 높아 LightGBM 선정
        + AUROC : 0.831, Training time : 40 seconds
    + 팀 내 업무
        + R을 이용한 데이터 전처리
        + 전처리한 데이터와 날씨 데이터 병합
        + Logistic regression, Xgboost
    + 분석 언어
        + R
        + Python
        + Jupyter Notebook
        + Excel
    + 새로운 시도 : 기존에 존재하는 데이터로는 예측이 힘들다고 판단
        + 따라서 공항별 날씨 데이터를 추가
            + 각 공항이 위치한 곳의 날씨를 기상청을 이용하여 데이터 수집
        + 하지만 공항의 이름이 각각 랜덤 처리 공항 식별 불가
        + 공항 별 비행기 운행 수 등의 데이터로 공항 식별 성공
        + 날씨 추가
    + 최종 프로젝트 pdf : 'https://github.com/Leejunho123/Project_2019/blob/main/Bigcon/Analysis_%ED%93%A8%EC%B2%98%EC%8A%A4%EB%A6%AC%EA%B7%B8_P.O.P_%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf'

