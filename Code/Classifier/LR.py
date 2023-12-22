import os
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import utils.tools as utils
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

os.chdir('C:\\Users\\Personal\\Desktop\\SVHEHS\\Dataset\\S.cerevisiae')
data_train = sio.loadmat(r'T_S_AC_11.mat')
data=data_train.get('data_AC')
pd.DataFrame(data)

row=data.shape[0]
column=data.shape[1]
index = [i for i in range(row)]
np.random.shuffle(index)
index=np.array(index)
data_=data[index,:]
shu=data_[:,np.array(range(1,column))]
label=data_[:,0]
X=shu
y=label


cv_clf =  LogisticRegression()
sepscores = []
skf= StratifiedKFold(n_splits=5)
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

for train, test in skf.split(X,y):
    X_train_enc=cv_clf.fit(X[train], y[train])
    y_score=cv_clf.predict_proba(X[test])
    yscore=np.vstack((yscore,y_score))
    y_test=utils.to_categorical(y[test])
    ytest=np.vstack((ytest,y_test))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= utils.categorical_probas_to_classes(y_score)
    y_test_tmp=y[test]
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('LR:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
result=np.mean(scores,axis=0)
HS=result.tolist()
sepscores.append(HS)
data_csv = pd.DataFrame(data=sepscores)
data_csv.to_csv('LR.csv')

row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore = pd.DataFrame(data=yscore)
yscore.to_csv('yscore_LR.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest = pd.DataFrame(data=ytest)
ytest.to_csv('ytest_LR.csv')