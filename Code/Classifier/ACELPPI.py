import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import utils.tools as utils
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier

os.chdir('C:\\Users\\Personal\\Desktop\\SVHEHS\\Dataset\\S.cerevisiae')
data_train = sio.loadmat(r'T_S_AC_11.mat')
data = data_train.get('data_AC')

row = data.shape[0]
column = data.shape[1]
index = [i for i in range(row)]
np.random.shuffle(index)
index = np.array(index)
data_ = data[index, :]
shu = data_[:, np.array(range(1, column))]
label = data_[:, 0]
train_data = shu
train_label = label


skf = StratifiedKFold(n_splits=5)
num_class = 2
sepscores = []
ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5

for train, test in skf.split(train_data, train_label):
    ET_model1 = ExtraTreesClassifier(n_estimators=500)
    lgb_model1 = lgb.LGBMClassifier(n_estimators=500)
    ET_model2 = ExtraTreesClassifier(n_estimators=500)
    lgb_model2 = lgb.LGBMClassifier(n_estimators=500)
    train_sets = []
    test_sets = []
    for clf in [ET_model1, lgb_model1, ET_model2, lgb_model2]:
        kf = KFold(n_splits=5)
        second_level_train_set = []
        test_nfolds_set = []
        for i, (train_index, test_index) in enumerate(kf.split(train_data[train])):
            x_tra, y_tra = train_data[train][train_index], train_label[train][train_index]
            x_tst, y_tst = train_data[train][test_index], train_label[train][test_index]
            clf.fit(x_tra, y_tra)
            second_level_train_ = clf.predict_proba(x_tst)
            second_level_train_set.append(second_level_train_)
            test_nfolds = clf.predict_proba(train_data[test])
            test_nfolds_set.append(test_nfolds)
        train_second = second_level_train_set
        train_second_level = np.concatenate(
            (train_second[0], train_second[1], train_second[2], train_second[3], train_second[4]), axis=0)
        test_second_level_ = np.array(test_nfolds_set)
        test_second_level = np.mean(test_second_level_, axis=0)
        train_set = train_second_level
        test_set = test_second_level
        train_sets.append(train_set)
        test_sets.append(test_set)
    meta_train = np.concatenate([result_set.reshape(-1, num_class) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1, num_class) for y_test_set in test_sets], axis=1)

    meta_testji = np.vstack((meta_test[:, 0], meta_test[:, 2], meta_test[:, 4], meta_test[:, 6]))
    meta_testou = np.vstack((meta_test[:, 1], meta_test[:, 3], meta_test[:, 5], meta_test[:, 7]))
    meta_testji1 = np.mean(meta_testji, axis=0)
    meta_testou1 = np.mean(meta_testou, axis=0)
    pre_score = np.vstack((meta_testji1, meta_testou1))
    pre_score = pre_score.T
    y_score = pre_score
    yscore = np.vstack((yscore, y_score))
    y_test = utils.to_categorical(train_label[test])
    ytest = np.vstack((ytest, y_test))
    fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)
    y_class = utils.categorical_probas_to_classes(y_score)
    y_test_tmp = train_label[test]
    acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])
    print('ACStackPPI:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc))
scores = np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100))
result = np.mean(scores, axis=0)
S = result.tolist()
sepscores.append(S)