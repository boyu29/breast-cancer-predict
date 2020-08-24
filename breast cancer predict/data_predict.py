import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

print(os.getcwd())

data = pd.read_csv('./data.csv')

data.drop("id",axis=1,inplace=True)
print(type(data))

print(data.head(5))

# feature selection
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']

X_train,X_test,y_train,y_test = train_test_split(data[features_remain],data['diagnosis'], 
                                                test_size=0.5,random_state=3,stratify=data['diagnosis'])
print(type(X_train))
print(X_train.head(5))
# train, test = train_test_split(data, test_size = 0.5,random_state=3,stratify=data['diagnosis'])
# X_train = train[features_remain]
# X_test = test[features_remain]
# y_train = train['diagnosis']
# y_test = test['diagnosis']

# data standardization
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# SVC Classifier
def SVM_Classifier(x_train, y_train):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(x_train, y_train)  
    return model

# def SVM_Classifier(x_train, y_train):
#     from sklearn.svm import LinearSVC
#     model = LinearSVC()
#     model.fit(x_train, y_train)  
#     return model

SVM_model = SVM_Classifier(X_train,y_train)
SVM_predict = SVM_model.predict(X_test)
print(SVM_predict)
SVC_accuracy = metrics.accuracy_score(y_test,SVM_predict)
print("SVC Accurary Is %.5f" %float(SVC_accuracy))

# Random Forest Classifier
def RF_Classifier(x_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(x_train, y_train)
    return model

RF_model = RF_Classifier(X_train,y_train)
RF_predict = RF_model.predict(X_test)
print(RF_predict)
RF_accuracy = metrics.accuracy_score(y_test, RF_predict)
print("Random Forest Accurary Is %.5f" %float(RF_accuracy))

# Logistic Regression Classifier
def LR_Classifier(x_train,y_train):
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2',max_iter=10000)  # 增加迭代次数:LogisticRegression里有一个max_iter（最大迭代次数）可以设置，默认为1000。
    model.fit(x_train, y_train)
    return model

LR_model = LR_Classifier(X_train,y_train)
LR_predict = LR_model.predict(X_test)
print(LR_predict)
LR_accuracy = metrics.accuracy_score(y_test, LR_predict)
print("Logistic Regression Accurary Is %.5f" %float(LR_accuracy))