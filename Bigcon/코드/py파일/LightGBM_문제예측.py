#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import lightgbm as lgb

#seed 고정하기
np.random.seed(0)

#data1이 저장 되어 있는 주소 입력하기
data=pd.read_csv('D:/data1.csv')

#범주형 변수를 카테고리화 하기
data.V1=data.V1.astype("category")
data.V2=data.V2.astype("category")
data.V3=data.V3.astype("category")
data.V4=data.V4.astype("category")
data.V5=data.V5.astype("category")
data.V6=data.V6.astype("category")
data.V7=data.V7.astype("category")
data.V8=data.V8.astype("category")

#종속변수와 변인 나눠주기.
x=data.drop('V8',axis=1)
y=data.V8

#여러가지 값을 대입해보고 AUC값이 높게 나오는 파라미터를 찾았다.
param = {'boosting_type': 'gbdt','num_leaves':700, 'objective':'binary','max_depth':200,'learning_rate':.05,'max_bin':255}
param['metric'] = ['auc', 'binary_logloss']
num_boost_round = 200
early_stopping_rounds = 10
nfold = 100
evals_result = {}
num_round = num_boost_round 

#data1을 7:3으로 train 데이터, test 데이터로 나누기
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
#x_test에는 실제 문제데이터중 data1의 모델을 사용할 데이터 불러오기
x_test=data=pd.read_csv('D:/AFSNT_DLY_data1.csv')

#범주형 변수를 카테고리화 하기
x_test.V1=x_test.V1.astype("category")
x_test.V2=x_test.V2.astype("category")
x_test.V3=x_test.V3.astype("category")
x_test.V4=x_test.V4.astype("category")
x_test.V5=x_test.V5.astype("category")
x_test.V6=x_test.V6.astype("category")
x_test.V7=x_test.V7.astype("category")

test_gbm=pd.DataFrame(x_train)
#현재 사용중인 PC에서 저장할 곳을 지정하기(이하 동일)
test_gbm.to_csv('D:/test_gbm.csv')
csv_data=pd.read_csv('D:/test_gbm.csv')
train_data=lgb.Dataset(x_train,label=y_train)
lgbm=lgb.train(param,train_data,num_round)
ypred2=lgbm.predict(x_test)

#data1 predprob값 저장하기
predprob4_gbm=pd.DataFrame(ypred2)
predprob4_gbm.to_csv('D:/predprob4_gbm.csv')

for j in range(0,len(ypred2)):
    if ypred2[j]>0.121:
        #type1 error와 type2 error의 비율이 거의 같게되는 cutoff값을 여러 값들을 대입해 보고 찾았다.
        ypred2[j]=1
    else:
        ypred2[j]=0

#data1 pred값 저장하기
pred4_gbm=pd.DataFrame(ypred2)
pred4_gbm.to_csv('D:/pred4_gbm.csv')



    

#seed 고정하기
np.random.seed(0)

#data2가 저장 되어 있는 주소 입력하기    
data=pd.read_csv('D:/data2.csv')

#범주형 변수를 카테고리화 하기
data.V1=data.V1.astype("category")
data.V2=data.V2.astype("category")
data.V3=data.V3.astype("category")
data.V4=data.V4.astype("category")
data.V5=data.V5.astype("category")
data.V6=data.V6.astype("category")
data.V7=data.V7.astype("category")
data.V8=data.V8.astype("category")

#종속변수와 변인 나눠주기.
x=data.drop('V8',axis=1)
y=data.V8

#여러가지 값을 대입해보고 AUC값이 높게 나오는 파라미터를 찾았다.
param = {'boosting_type': 'gbdt','num_leaves':700, 'objective':'binary','max_depth':200,'learning_rate':.05,'max_bin':255}
param['metric'] = ['auc', 'binary_logloss']
num_boost_round = 200
early_stopping_rounds = 10
nfold = 100
evals_result = {}
num_round = num_boost_round 

#data2를 7:3으로 train 데이터, test 데이터로 나누기
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
#x_test에는 실제 문제데이터중 data2의 모델을 사용할 데이터 불러오기
x_test=data=pd.read_csv('D:/AFSNT_DLY_data2.csv')

#범주형 변수를 카테고리화 하기
x_test.V1=x_test.V1.astype("category")
x_test.V2=x_test.V2.astype("category")
x_test.V3=x_test.V3.astype("category")
x_test.V4=x_test.V4.astype("category")
x_test.V5=x_test.V5.astype("category")
x_test.V6=x_test.V6.astype("category")
x_test.V7=x_test.V7.astype("category")

test_gbm=pd.DataFrame(x_train)
test_gbm.to_csv('D:/test-2_gbm.csv')
testarp_gbm=pd.DataFrame(x_test)
testarp_gbm.to_csv('D:/testarp_gbm.csv')
csv_data=pd.read_csv('D:/test-2_gbm.csv')
train_data=lgb.Dataset(x_train,label=y_train)
lgbm=lgb.train(param,train_data,num_round)
ypred2=lgbm.predict(x_test)

#data2 predprob값 저장하기
predprob4_gbm=pd.DataFrame(ypred2)
predprob4_gbm.to_csv('D:/predprob4-2_gbm.csv')

for j in range(0,len(ypred2)):
    if ypred2[j]>0.1235:
        #type1 error와 type2 error의 비율이 거의 같게되는 cutoff값을 여러 값들을 대입해 보고 찾았다.
        ypred2[j]=1
    else:
        ypred2[j]=0

#data2 pred값 저장하기
pred4_gbm=pd.DataFrame(ypred2)
pred4_gbm.to_csv('D:/pred4-2_gbm.csv')





#data1과 data2의 결과를 합치기 위해 만들어준 csv파일 불러오기
predprob4_gbm=pd.read_csv('D:/predprob4_gbm.csv')
pred4_gbm=pd.read_csv('D:/pred4_gbm.csv')

predprob4_2_gbm=pd.read_csv('D:/predprob4-2_gbm.csv')
pred4_2_gbm=pd.read_csv('D:/pred4-2_gbm.csv')
testarp_gbm=pd.read_csv('D:/testarp_gbm.csv')

#'V5'가 5(ARP13)와 6(ARP14)인 경우의 행값을 뽑아내기
a=list(testarp_gbm.columns.values)
b=a[0:4]
c=a[6:24]
testarp_gbm.drop(b,axis=1,inplace=True)
testarp_gbm.drop(c,axis=1,inplace=True)
k=[]
d=len(testarp_gbm)
for i in range(0,d):
    if testarp_gbm['V5'][i]==5 or testarp_gbm['V5'][i]==6:
        k.append(i)

predprob4_2_arp1314=pd.DataFrame(predprob4_2_gbm['0'][k])
predprob4_2_arp1314.to_csv('D:/predprob4-2(arp13,arp14)_gbm.csv')

pred4_2_arp1314=pd.DataFrame(pred4_2_gbm['0'][k])
pred4_2_arp1314.to_csv('D:/pred4-2(arp13,arp14)_gbm.csv')


# 코드를 실행하여 작성된 pred4_gbm, predprob4_gbm, pred4-2(arp13,arp14)_gbm, predprob4-2(arp13,arp14)_gbm, 4개 csv파일로 AFSNT_DLY.csv를 작성함.
#1. myaddin 프로그램을 설치한다.
#2. myaddin 프로그램을 실행하여 매크로를 허용한다.
#3, AFSNT_DLY.csv에서 공항에 필터를 걸어 ARP13, ARP14를 제외한다.
#4. Ctrl+Alt+C를 눌러 pred4_gbm.csv파일에서 값을 복사한다.
#5. Ctrl+Alt+V를 눌러 AFSNT_DLY.csv파일에 DLY에 값을 붙여넣는다.
#6. Ctrl+Alt+C를 눌러 predprob4_gbm.csv파일에서 값을 복사한다.
#7. Ctrl+Alt+V를 눌러 AFSNT_DLY.csv파일에 DLY_RATE에 값을 붙여넣는다.
#8. AFSNT_DLY.csv에서 공항에 필터를 걸어 ARP13, ARP14만 나오게 한다..
#9. Ctrl+Alt+C를 눌러 pred4-2(arp13,arp14)_gbm.csv파일에서 값을 복사한다.
#10. Ctrl+Alt+V를 눌러 AFSNT_DLY.csv파일에 DLY에 값을 붙여넣는다.
#11. Ctrl+Alt+C를 눌러 predprob4-2(arp13,arp14)_gbm.csv파일에서 값을 복사한다.
#12. Ctrl+Alt+V를 눌러 AFSNT_DLY.csv파일에 DLY_RATE에 값을 붙여넣는다.
#13. DLY_RATE을 엑셀 함수를 이용하여 소수점 셋째자리에서 반올림한다.

