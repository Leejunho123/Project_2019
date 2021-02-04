## 빅콘테스트 Analysis 퓨처스리그
- Contents
    + 빅콘테스트 소개자료 : 빅콘테스트 퓨처스리그 설명자료.pptx
    + 공항별, 항공기별 데이터를 이용하여 항공기 지연 예측
    + 기존에 존재하는 데이터로는 예측이 힘들다고 판단
        + 따라서 공항별 날씨 데이터를 추가
        + 하지만 공항의 이름이 각각 랜덤 처리 공항 식별 불가
        + 공항 별 비행기 운행 수 등의 데이터로 공항 식별 성공
        + 날씨 추가
    + 사용한 모델
        1. Logistic regression
        2. Xgboost
        3. LightGBM
        + xgboost 와 LightGBM 의 AUROC값이 비슷했으나 LightGBM의 속도가 압도적으로 높아 LightGBM 선정
        + AUROC : 0.831, Training time : 40 seconds
    + 최종 프로젝트 pdf : 'https://github.com/Leejunho123/Project_2019/blob/main/Bigcon/Analysis_%ED%93%A8%EC%B2%98%EC%8A%A4%EB%A6%AC%EA%B7%B8_P.O.P_%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf'

