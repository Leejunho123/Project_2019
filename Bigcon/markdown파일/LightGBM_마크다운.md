
V1~V8을 범주형 변수로 취급하기 위해 카테고리화 시켰다.

data를 7:3으로 train 데이터, test 데이터로 나눠서 예측모델을 만들고 거기에 맞는 parameter를 설정하였기 때문에 문제데이터 에서 예측을 할 때에도 data를 7:3으로 train 데이터, test 데이터로 나누고 같은 값들을 설정해서 예측모델을 만들었다.

type1 error와 type2 error의 비율이 거의 같게되는 cutoff값을 여러 값들을 대입해 보고 찾았다.

ARP13과 ARP14의 경우에는 항공기상 데이터가 없다는 점이 다른 공항과 다르기 때문에 예측모델을 따로 쓸 필요성이 있었다. 따라서 항공기상 데이터를 제외하고 예측하는 모델을 하나 더 만들어서 ARP13과 ARP14를 예측하는데 사용한다.
