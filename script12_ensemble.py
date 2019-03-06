# libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
%matplotlib inline

# read data
df = pd.read_csv('data07_diabetes.csv')
X = df.iloc[:,:-1]
y = df['Y']
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.5,random_state=1)

###########################################################
# Bagging Methods
###########################################################

from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)

# usual knn
knn.fit(xtrain,ytrain)
print(knn.score(xtrain,ytrain),knn.score(xtest,ytest))

# full bagging n_estimators : 모델의 갯수, max_samples : resample의 비율, 0.5로 하면 반만 중복허용
# max_features : feature를 뽑을 때 중복허용 비율 bootstrap : true(중복허용) bootstrap_features : true(중복허용)
bf = BaggingRegressor(knn,n_estimators=100,max_samples=1.0,max_features=1.0,random_state=0)
bf.fit(xtrain,ytrain)
print(bf.score(xtrain,ytrain),bf.score(xtest,ytest))

# bagging with subsampling and feature randomization
bf = BaggingRegressor(knn,n_estimators=500,max_samples=0.5,max_features=0.5)
bf.fit(xtrain,ytrain)
print(bf.score(xtrain,ytrain),bf.score(xtest,ytest))

# effect of estimators
np.random.seed(0)
n_list = [1,5,10,20,30,50,100,200,500,1000]
s = np.zeros((len(n_list),2))
for i in range(len(n_list)):
    bf = BaggingRegressor(knn,n_estimators=n_list[i],max_samples=0.5,max_features=0.5)
    bf.fit(xtrain,ytrain)
    s[i,0] = bf.score(xtrain,ytrain)
    s[i,1] = bf.score(xtest,ytest)
s
plt.plot(np.log10(n_list),s,marker='o')

# parameter tunning
np.random.seed(1)
params = np.arange(2,20,2)
s = np.zeros((len(params),3))
for i in range(len(params)):
    f = BaggingRegressor(KNeighborsRegressor(params[i]),n_estimators=200,
                          random_state=0,max_samples=0.5,max_features=0.5,oob_score=True)
    f.fit(xtrain,ytrain)
    s[i,0] = f.score(xtrain,ytrain)
    s[i,1] = cross_val_score(f,xtrain,ytrain,cv=5).mean()
#    s[i,1] = f.oob_score_   
#oob_score가 더 빠른 출력 가능하다. CV하려면 많은 computing power 소모
    s[i,2] = f.score(xtest,ytest)
s
plt.plot(s)


###########################################################
# Random Forest
###########################################################

# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,random_state=0)
rf.fit(xtrain,ytrain)
yhat_test = rf.predict(xtest)
print(rf.score(xtrain,ytrain),rf.score(xtest,ytest))

# number of features to be chosen
rf = RandomForestRegressor(n_estimators=100,random_state=0,max_features='sqrt')
rf.fit(xtrain,ytrain)
print(rf.score(xtrain,ytrain),rf.score(xtest,ytest))

# tree size
rf = RandomForestRegressor(n_estimators=100,random_state=0,
                           max_features='sqrt',max_leaf_nodes=100)
rf.fit(xtrain,ytrain)
print(rf.score(xtrain,ytrain),rf.score(xtest,ytest))

# oob score
rf = RandomForestRegressor(n_estimators=100,random_state=0,
                           max_features='sqrt',max_leaf_nodes=100,
                           oob_score=True)
rf.fit(xtrain,ytrain)
print(rf.score(xtrain,ytrain),rf.score(xtest,ytest),rf.oob_score_)  # out-of-bag score

# parameter tuning using oob scores
tree_size = np.arange(2,50,2)
s = np.zeros((len(tree_size),4))
for i in range(len(tree_size)):
    rf = RandomForestRegressor(n_estimators=100,random_state=0,
                           max_features='sqrt',max_leaf_nodes=tree_size[i],
                           oob_score=True)
    rf.fit(xtrain,ytrain)
    s[i,0] = rf.score(xtrain,ytrain)
    s[i,1] = rf.oob_score_ 
    s[i,2] = cross_val_score(rf,xtrain,ytrain,cv=5).mean()
    s[i,3] = rf.score(xtest,ytest)

plt.plot(tree_size,s)


###########################################################
# Gradient Boosted Tree
###########################################################

# gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(learning_rate=0.1,n_estimators=100,max_depth=3)
gb.fit(xtrain,ytrain)
yhat_test = gb.predict(xtest)
print(gb.score(xtrain,ytrain),gb.score(xtest,ytest))

# number of estimators
n_list = np.arange(2,200,2)
s = np.zeros((len(n_list),3))
for i in range(len(n_list)):
    gb = GradientBoostingRegressor(learning_rate=0.1,
                                   n_estimators=n_list[i],max_depth=3)
    #learning rate : 람다 shrunkage index
    
    gb.fit(xtrain,ytrain)
    s[i,0] = gb.score(xtrain,ytrain)
    s[i,1] = cross_val_score(gb,xtrain,ytrain,cv=5).mean() 
    s[i,2] = gb.score(xtest,ytest)

plt.plot(n_list,s)

# oob improvement
gb = GradientBoostingRegressor(learning_rate=0.1,n_estimators=100,
                                max_depth=3,subsample=0.5)
gb.fit(xtrain,ytrain)
plt.plot(gb.oob_improvement_)

idx = (gb.oob_improvement_<0).tolist().index(True)
gb = GradientBoostingRegressor(learning_rate=0.1,n_estimators=idx+1,
                                max_depth=3,subsample=0.5,random_state=0)
gb.fit(xtrain,ytrain)
gb.score(xtest,ytest)


###########################################################
# Practice
###########################################################

# read data
df = pd.read_csv('data07_diabetes.csv')
df_data = df.iloc[:,:-1]
df_target = (df['Y']>140).factorize()[0]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(
        df_data,df_target,test_size=0.33,random_state=0)

# bagging knn
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier


# random forest
from sklearn.ensemble import RandomForestClassifier


# gradient boosting tree
from sklearn.ensemble import GradientBoostingClassifier







# PLEASE DO NOT GO DOWN BEFORE YOU TRY BY YOURSELF

###########################################################
# Practice Reference Code
###########################################################

# read data
df = pd.read_csv('data07_diabetes.csv')
df_data = df.iloc[:,:-1]
df_target = (df['Y']>140).factorize()[0]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(
        df_data,df_target,test_size=0.33,random_state=0)

# bagging knn
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
params = np.arange(3,20,2)
s = np.zeros((len(params),3))
for i in range(len(params)):
    f = BaggingClassifier(KNeighborsClassifier(params[i]),n_estimators=100,
                          random_state=0,max_samples=0.5,max_features=0.5,
                          oob_score=True)
    f.fit(xtrain,ytrain)
    s[i,0] = f.score(xtrain,ytrain)
    #s[i,1] = cross_val_score(f,xtrain,ytrain,cv=5).mean()
    s[i,1] = f.oob_score_    
    s[i,2] = f.score(xtest,ytest)
score_bagknn = s[np.argmax(s[:,1]),2]

# random forest
from sklearn.ensemble import RandomForestClassifier
tree_size = np.arange(2,30,2)
s = np.zeros((len(tree_size),3))
for i in range(len(tree_size)):
    rf = RandomForestClassifier(n_estimators=100,random_state=0,
                           max_features='sqrt', max_leaf_nodes=tree_size[i],
                           oob_score=True)
    rf.fit(xtrain,ytrain)
    s[i,0] = rf.score(xtrain,ytrain)
    s[i,1] = rf.oob_score_ 
    s[i,2] = rf.score(xtest,ytest)
score_rf = s[np.argmax(s[:,1]),2]

# gradient boosting tree
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(learning_rate=0.01,n_estimators=500,
                                max_depth=1,subsample=0.5,random_state=0)
gb.fit(xtrain,ytrain)
plt.plot(gb.oob_improvement_)

idx = (gb.oob_improvement_<0).tolist().index(True)
gb = GradientBoostingClassifier(learning_rate=0.01,n_estimators=idx+1,
                                max_depth=1,subsample=0.5,random_state=0)
gb.fit(xtrain,ytrain)
score_gb = gb.score(xtest,ytest)

