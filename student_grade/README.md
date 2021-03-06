## 설문조사를 통한 성적에 미치는 요인들 분석
- Contents
    + 목표 : 설문 조사를 통한 포르투갈 중등학교 학생의 성적에 미치는 변수 연구 및 성적 예측
    + 대학 과목 내 프로젝트 : 데이터 마이닝 및 실험
    + 기간 : 2018년 5월 ~ 2018년 6월 (1개월)
    + 팀 인원 : 3
    + 사용된 분석
        1. GLM
        2. Ridge, Lasso
        3. GAM
        4. SVM
        5. Decision Tree
        6. Random Forest
        7. Bagging
        8. Gradient Boosting
        9. Ada Boosting
        10. XG Boosting
    + 팀 내 업무 : GLM, Ridge, Lasso, GAM, SVM, XG Boosting, 최종 모델 선정을 위한 전체 for문 작성
    + 사용 언어 : R
    + 특이점 
        1. 각 모델들의 변동성이 커서 특정 모델을 선정하기에 어려움이 있었음
        2. 따라서 모든 모델을 100번 돌려 AUC의 평균값을 도출
        3. 이 평균 값을 이용하여 최종 모델 선정
    + 최종 예측률 : 0.7095    
    + 데이터 및 코드는 소실
    + pdf : https://github.com/Leejunho123/Project_2019/blob/main/student_grade/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A7%88%EC%9D%B4%EB%8B%9D%20PPT%20%EC%B5%9C%EC%A2%85.pdf