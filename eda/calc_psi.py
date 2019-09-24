import pandas as pd
import numpy as np


def calc_psi(train, test, features=None, cate_cols=None, n=10):
    """
    全量计算特征的psi
    :param train:
    :param test:
    :param features: 全部需要计算的特征，包括类别型和连续型
    :param cate_cols: 类别型特征
    :param n:
    :return:
    """
    if features is None:
        features = train.columns.tolist()
    train = train.copy().fillna(-999)
    test = test.copy().fillna(-999)
    train['group'] = 'train'
    test['group'] = 'test'
    data = pd.concat([train, test], axis=0)
    psi = {}

    if cate_cols is None:
        cate_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        cate_cols = None if len(cate_cols) == 0 else cate_cols

    if cate_cols is None:
        cut_off_p = np.linspace(0, 1, n + 1)
        for x in features:
            cut_off = sorted(train[x].quantile(cut_off_p).unique())
            cut_off[0] = -np.inf
            cut_off[-1] = np.inf
            print('=== cut_off of ', x, ' ===')
            print(cut_off)
            data[x] = pd.cut(data[x], cut_off)
            res = data.groupby([x, 'group']).size().unstack().fillna(1)
            res = res / np.sum(res, axis=0)
            print('=== describe of data ===')
            print(res)
            psi_x = np.sum((res['train'] - res['test']) * np.log(res['train'] / res['test']))
            print('===psi of ', x, ' :', psi_x)
            psi[x] = np.round(psi_x, 4)
    else:
        part_features = [x for x in features if x not in cate_cols]
        cut_off_p = np.linspace(0, 1, n + 1)
        for x in part_features:
            cut_off = sorted(train[x].quantile(cut_off_p).unique())
            cut_off[0] = -np.inf
            cut_off[-1] = np.inf
            data[x] = pd.cut(data[x], cut_off)
            print('=== cut_off of ', x, ' ===')
            print(cut_off)
            res = data.groupby([x, 'group']).size().unstack().fillna(1)
            res = res / np.sum(res, axis=0)
            print('=== describe of data ===')
            print(res)
            psi_x = np.sum((res['train'] - res['test']) * np.log(res['train'] / res['test']))
            psi[x] = np.round(psi_x, 4)
        for x in cate_cols:
            res = data.groupby([x, 'group']).size().unstack().fillna(1)
            res = res / np.sum(res, axis=0)
            psi_x = np.sum((res['train'] - res['test']) * np.log(res['train'] / res['test']))
            psi[x] = np.round(psi_x, 4)
    psi_all = pd.DataFrame(columns=['name', 'psi'])
    psi_all['name'] = psi.keys()
    psi_all['psi'] = psi.values()
    return psi_all
