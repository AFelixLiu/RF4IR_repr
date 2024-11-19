"""
# Example code for the "Machine Learning interpretation of the correlation between
# infrared emission features of interstellar polycyclic aromatic hydrocarbons" paper.

# The following are the required input files.
# Bins.json : Spectral histogram width wavenumber list information.
# Fingerprint.json : Molecular fingerprint information dataset. (2926*19794)
# ID_list.pck : Molecular ID information.
# IRdata.json : Molecular label dataset. (2925)
"""

# -*- coding: utf-8 -*-
try:
    from sklearn.ensemble import RandomForestRegressor
    import json
    import pickle as pck
    import numpy as np
    import copy
    import os
except ModuleNotFoundError:
    raise Exception("Please make sure sklearn, json, pickle and numpy are installed!")


# def findKey(Dict, yourValue):
#     f_key = [k for k, v in Dict.items() if v == yourValue]
#     return f_key


def write_out(file, Y_test, Y_pred, RMSE, EMD, t_waverange):
    fw = open(file, 'w')
    fw.write(f"RMSE: {RMSE}\t EMD: {EMD}\n")
    fw.write("------*------*------*------*------*------*------\n")
    fw.write('wavenumber(/cm)\tY_pred(km/mol)\tY_test(km/mol)\n')
    for i in range(len(Y_pred)):
        if t_waverange == 'low':
            fw.write(str((bins[i] + bins[i + 1]) / 2) + '\t' + str(Y_pred[i]) + '\t' + str(Y_test[i]) + '\n')
        elif t_waverange == 'high':
            fw.write(str((bins[bin_low + i] + bins[bin_low + i + 1]) / 2) + '\t' + str(Y_pred[i]) + '\t' + str(Y_test[i]) + '\n')
    fw.close()


def calcError(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    Error = {}
    a = y_pred / np.sum(y_pred)
    b = y_test / np.sum(y_test)
    EMD = np.sum(np.abs(np.cumsum(a - b)))
    Error['EMD'] = EMD
    RMSE = (np.sum((y_test - y_pred) ** 2) / len([*y_test])) ** 0.5
    Error['RMSE'] = RMSE
    return Error


def MR_RF(t_waverange, t_modeltype):
    X_train, X_test, Y_train, Y_test, IDs_split = build_data(IDs_train, IDs_test, t_modeltype)
    Y_train = Normalized(Y_train)
    Y_test = Normalized(Y_test)

    print('train size:', X_train.shape, Y_train.shape, len(IDs_split['train']))
    print('test size:', X_test.shape, Y_test.shape, len(IDs_split['test']))

    rfr = RFModel()
    rfr = rfr.fit(X_train, Y_train)
    print("train loss | mse: " + str(rfr.oob_score_))
    Y_pred = rfr.predict(X_test)
    Y_pred = Normalized(Y_pred)

    predData = {}
    mean_EMD, mean_RMSE = 0, 0
    for i in range(Y_test.shape[0]):
        y_test = Y_test.tolist()[i]
        y_pred = Y_pred.tolist()[i]
        Error = calcError(y_test, y_pred)
        mean_EMD += Error['EMD']
        mean_RMSE += Error['RMSE']
        pred = {'Y_test': copy.deepcopy(y_test), 'Y_pred': copy.deepcopy(y_pred),
                'EMD': Error['EMD'], 'RMSE': Error['RMSE']}
        predData[IDs_split['test'][i]] = copy.deepcopy(pred)

    mean_EMD /= Y_test.shape[0]
    mean_RMSE /= Y_test.shape[0]
    print("test loss | meanEMD: " + str(mean_EMD) + ", meanRMSE: " + str(mean_RMSE))

    for i in range(len(IDs_split['test'])):
        k = IDs_split['test'][i]
        y_test = predData[k]['Y_test']
        y_pred = predData[k]['Y_pred']
        RMSE = predData[k]['RMSE']
        EMD = predData[k]['EMD']
        res_file = res_path + k + '_' + t_waverange + '_' + t_modeltype + '.txt'
        write_out(res_file, y_test, y_pred, RMSE, EMD, t_waverange)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def RFModel():
    rfr = RandomForestRegressor(bootstrap=True,
                                # criterion='mse',
                                criterion='friedman_mse',
                                max_depth=None,
                                max_features='sqrt',
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_samples_leaf=1,
                                min_samples_split=2,
                                min_weight_fraction_leaf=0.0,
                                n_estimators=1000,
                                n_jobs=None,
                                oob_score=True,
                                random_state=52,
                                verbose=0,
                                warm_start=False)
    return rfr


def Normalized(Y):
    for i in range(Y.shape[0]):
        if Y.shape[1] > 1:
            Y[i] = Y[i] / np.max(Y[i])
        else:
            Y[i] = Y[i] / np.max(Y)
    Y_new = Y
    return Y_new


def build_data(t_IDs_train, t_IDs_test, t_modeltype):
    IDs_split = {'train': t_IDs_train, 'test': t_IDs_test}
    X_train, X_test = None, None
    Y_train = np.array([IRdata[k] for k in IDs_split['train']])
    Y_test = np.array([IRdata[k] for k in IDs_split['test']])

    if t_modeltype == 'ECFP':
        X_tra = []
        for id_ in IDs_split['train']:
            fp = []
            for idx in feature_indexes:
                fp.append(Fingerprint[id_][idx])
            X_tra.append(copy.deepcopy(fp))

        X_tes = []
        for id_ in IDs_split['test']:
            fp = []
            for idx in feature_indexes:
                fp.append(Fingerprint[id_][idx])
            X_tes.append(copy.deepcopy(fp))

        X_train = np.array(X_tra)
        X_test = np.array(X_tes)

    return X_train, X_test, Y_train, Y_test, IDs_split


def calculate_feature_importance(t_modeltype):
    X_train, X_test, Y_train, Y_test, IDs_split = build_data(IDs_train, IDs_test, t_modeltype)
    Y_train = Normalized(Y_train)
    Y_test = Normalized(Y_test)

    print('train size:', X_train.shape, Y_train.shape, len(IDs_split['train']))
    print('test size:', X_test.shape, Y_test.shape, len(IDs_split['test']))

    rfr = RFModel()
    """
    挑选重要特征：
    1 首先使用所有特征训练
    2 然后将特征按照重要性排序进行选择
    """
    feature_importance = rfr.fit(X_train, Y_train).feature_importances_  # 使用所有特征训练，并获得对应特征的重要性值
    feature_importance = 100.0 * (feature_importance / feature_importance.max())  # 对重要性进行归一化
    t_feature_idx = np.argsort(feature_importance)  # 对数组进行排序，并返回排序后的索引值数组
    t_feature_idx = t_feature_idx[::-1]  # 数组反转
    # pos = np.arange(t_sorted_idx.shape[0]) + 0.5
    return t_feature_idx


def splitList(List, splits):
    indexes = np.random.permutation(len(List))
    groups = {}
    ratio = 0.0
    for i in range(len(splits)):
        ratio += splits[i]
        if i != len(splits) - 1:
            ids = indexes[int(len(indexes) * (ratio - splits[i])):int(len(indexes) * ratio)].tolist()
            groups[i] = [List[k] for k in ids]
        else:
            ids = indexes[int(len(indexes) * (ratio - splits[i])):].tolist()
            groups[i] = [List[k] for k in ids]
    return groups


def LowOrHigh(t_waverange, t_ID_list, t_IR_data):
    IDs_screen = []
    IR_screen = {}
    if t_waverange == 'high':
        for i in t_ID_list:
            sum_ir = sum(t_IR_data[i][bin_low:bin_high])
            if sum_ir > 0.0:
                IDs_screen.append(i)
                IR_screen[i] = t_IR_data[i][bin_low:bin_high]
    elif t_waverange == 'low':
        for i in t_ID_list:
            IDs_screen.append(i)
            IR_screen[i] = t_IR_data[i][bin_low:bin_high]
    return IDs_screen, IR_screen


def IR_bin_cut(t_waverange):
    t_bin_low, t_bin_high = 0, 0
    if t_waverange == 'low':
        t_bin_low = 0
        t_bin_high = 137
    elif t_waverange == 'high':
        t_bin_low = 137
        t_bin_high = 225
    return t_bin_low, t_bin_high


if __name__ == '__main__':
    Bins = json.load(open("Bins.json", "r"))
    bins = Bins['bins']
    ID_list = pck.load(open("ID_list.pck", 'rb'))
    Fingerprint = json.load(open('Fingerprint.json', "r"))
    IR_data = json.load(open("IRdata.json", "r"))

    waverange = 'low'  # low/high
    modeltypelist = ['ECFP']
    bin_low, bin_high = IR_bin_cut(waverange)
    IDs, IRdata = LowOrHigh(waverange, ID_list, IR_data)
    print('IDsSize:', len(IDs))

    for modeltype in modeltypelist:
        print('---current modeltype>>>', modeltype)

        if modeltype == 'ECFP':
            feature_indexes = list(range(len(Fingerprint[IDs[0]])))

        IDs_groups = splitList(IDs, [0.2, 0.2, 0.2, 0.2, 0.2])

        IDs_train = [k for k in IDs if k not in IDs_groups[0]]
        IDs_test = [k for k in IDs if k in IDs_groups[0]]

        feature_idx = None
        if modeltype in ['ECFP']:
            print("calculating feature importance...")
            feature_idx = calculate_feature_importance(modeltype)

        if modeltype == 'ECFP':
            feature_indexes = feature_idx[:1000]  # 选择重要性值在前1000的特征（ECFP描述符共有19794个特征）

        print("------*------*------*------*------*------*------")
        print("training model...")
        for t in range(0, 1):
            res_path = './results' + '/result_' + str(t + 1) + '/'
            mkdir(res_path)
            IDs_groups = splitList(IDs, [0.2, 0.2, 0.2, 0.2, 0.2])
            for i, group in IDs_groups.items():
                IDs_train = [k for k in IDs if k not in group]
                IDs_test = [k for k in IDs if k in group]
                print(f"Round {i + 1}/5")
                if len(IDs_test) > 0:
                    MR_RF(waverange, modeltype)
    print('all done !!!')
