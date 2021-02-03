#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import time
from matplotlib import pyplot as plt
#plot이 나오지 않을경우 한번 더 코드 실행

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
param = {'boosting_type': 'gbdt','num_leaves':700, 'objective':'binary','max_depth':200,'learning_rate':.05,'max_bin':255,'importance_type':'gain'}
param['metric'] = ['auc', 'binary_logloss']
num_boost_round = 200
early_stopping_rounds = 10
nfold = 100
evals_result = {}
num_round = num_boost_round 

#data1을 7:3으로 train 데이터, test 데이터로 나누기
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
test4_gbm=pd.DataFrame(y_test)
#현재 사용중인 PC에서 저장할 곳을 지정하기(이하 동일)
test4_gbm.to_csv('D:/test4_gbm.csv')
test_gbm=pd.DataFrame(x_train)
test_gbm.to_csv('D:/test_gbm.csv')
csv_data=pd.read_csv('D:/test_gbm.csv')
train_data=lgb.Dataset(x_train,label=y_train)

#importance plot에 사용할 변수명 만들어주기.
xname=['V1','V2','V3','V4','V5','V6','V7','T1','T2','W1','W2','W3','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12']

#모델링 소요시간 구하기
start = time.time()
lgbm=lgb.train(param,train_data,num_round,feature_name=[i for i in xname])
print("time :", time.time() - start)

ypred2=lgbm.predict(x_test)
#data1 predprob값 저장하기
predprob4_gbm=pd.DataFrame(ypred2)
predprob4_gbm.to_csv('D:/predprob4_gbm.csv')

#data1 lightGBM AUC
print("data1 AUC")
print(roc_auc_score(y_test,ypred2))

for i in range(0,1):
    for j in range(0,len(ypred2)):
        #type1 error와 type2 error의 비율이 거의 같게되는 cutoff값을 여러 값들을 대입해 보고 찾았다.
        if ypred2[j]>0.121:
            ypred2[j]=1
        else:
            ypred2[j]=0

#data1 pred값 저장하기
pred4_gbm=pd.DataFrame(ypred2)
pred4_gbm.to_csv('D:/pred4_gbm.csv')


#data1 lightGBM importance plot
print('Plotting feature importances...')
ax = lgb.plot_importance(lgbm, max_num_features=20)
plt.show()

    
#seed 고정하기
np.random.seed(0)

#data2가 저장되어있는 주소 입력하기.    
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
param = {'boosting_type': 'gbdt','num_leaves':700, 'objective':'binary','max_depth':200,'learning_rate':.05,'max_bin':255,'importance_type':'gain'}
param['metric'] = ['auc', 'binary_logloss']
num_boost_round = 200
early_stopping_rounds = 10
nfold = 100
evals_result = {}
num_round = num_boost_round 

#data2를 7:3으로 train 데이터, test 데이터로 나누기
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
test4_gbm=pd.DataFrame(y_test)
test4_gbm.to_csv('D:/test4-2_gbm.csv')
test_gbm=pd.DataFrame(x_train)
test_gbm.to_csv('D:/test-2_gbm.csv')
testarp_gbm=pd.DataFrame(x_test)
testarp_gbm.to_csv('D:/testarp_gbm.csv')
csv_data=pd.read_csv('D:/test-2_gbm.csv')
train_data=lgb.Dataset(x_train,label=y_train)

#importance plot에 사용할 변수명 만들어주기.
xname=['V1','V2','V3','V4','V5','V6','V7','T1','T2','W1','W2','W3']
lgbm=lgb.train(param,train_data,num_round,feature_name=[i for i in xname])
ypred2=lgbm.predict(x_test)
#data2 predprob값 저장하기
predprob4_gbm=pd.DataFrame(ypred2)
predprob4_gbm.to_csv('D:/predprob4-2_gbm.csv')

#data2 lightGBM AUC
print("data2 AUC")
print(roc_auc_score(y_test,ypred2))


for i in range(0,1):
    for j in range(0,len(ypred2)):
        if ypred2[j]>0.1235:
            #type1 error와 type2 error의 비율이 거의 같게되는 cutoff값을 여러 값들을 대입해 보고 찾았다.
            ypred2[j]=1
        else:
            ypred2[j]=0

#data2 pred값 저장하기
pred4_gbm=pd.DataFrame(ypred2)
pred4_gbm.to_csv('D:/pred4-2_gbm.csv')

#data2 lightGBM importance plot
print('Plotting feature importances...')
ax = lgb.plot_importance(lgbm, max_num_features=20)
plt.show()

    
#data1과 data2의 결과를 합치기 위해 만들어준 csv파일 불러오기
predprob4_gbm=pd.read_csv('D:/predprob4_gbm.csv')
test4_gbm=pd.read_csv('D:/test4_gbm.csv')
pred4_gbm=pd.read_csv('D:/pred4_gbm.csv')

predprob4_2_gbm=pd.read_csv('D:/predprob4-2_gbm.csv')
test4_2_gbm=pd.read_csv('D:/test4-2_gbm.csv')
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


all_test=pd.concat([test4_gbm['V8'],test4_2_gbm['V8'][k]])
all_predprob=pd.concat([predprob4_gbm['0'],predprob4_2_gbm['0'][k]])
all_pred=pd.concat([pred4_gbm['0'],pred4_2_gbm['0'][k]])

#data1&data2 lightGBM AUC
print("data1&data2 AUC")
print(roc_auc_score(all_test,all_predprob))

#data1&data2 lightGBM Crosstab
print("Crosstab")
print(pd.crosstab(all_test,all_pred))

result1=(pd.crosstab(all_test,all_pred)[0][0])/((pd.crosstab(all_test,all_pred)[1][0])+(pd.crosstab(all_test,all_pred)[0][0]))

result2=(pd.crosstab(all_test,all_pred)[1][1])/((pd.crosstab(all_test,all_pred)[0][1])+(pd.crosstab(all_test,all_pred)[1][1]))

print("실제가 0인데 0으로 예측한 비율")     
print(result1)
print("실제가 1인데 1로 예측한 비율")     
print(result2)

def rocvis(true , prob , label ) :
    if type(true[0]) == str :
        le = LabelEncoder()
        true = le.fit_transform(true)
    else :
        pass
    fpr, tpr, thresholds = roc_curve(true, prob)
    plt.plot(fpr, tpr, marker='.', label = label  )

#data1&data2 lightGBM Roc Curve    
plt.plot([0, 1], [0, 1], linestyle='--')
rocvis(all_test , all_predprob , "LightGBM")
plt.legend(fontsize = 18)
plt.title("Roc Curve" , fontsize= 25)
plt.show()

