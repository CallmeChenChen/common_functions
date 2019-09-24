import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def stacking_diagram(data, features, label='label', decimals=2, ylim_min=0, ylim_max=1, fill_na=-999, save_fig=False):
    """
    特征(连续)每个分位点，label 0和1的占比情况(也试用于多分类情形)。主要关注的特征的趋势是否符合业务逻辑。
    缺点：数据太少，分位点数据较少，0和1比具有一定的偶然性。需要提前处理
    :param data: 标签数据
    :param features: 连续型特征
    :param label: 实际标签 'label' 或 'y ...
    :param decimals: 1代表以10为间隔的分位点(可以看大致趋势)
    :param ylim_min:
    :param ylim_max:
    :param fill_na:
    :param save_fig:
    :return:
    """

    data = data.copy()
    data[features] = data[features].fillna(fill_na)
    data[features] =np.around(data[features].rank(method='max', pct=True), decimals)
    result = data.groupby([features,label]).size().unstack().fillna(1)
    start_result = result.head(1)
    start_result.index=[0.0,]
    result = pd.concat([start_result,result],axis=0)
    code_start ="plt.stackplot(result.index,"
    cols ="labels = ["
    label_unique = list(data[label].unique())
    label_unique.sort()
    for i in label_unique:
        code_start = code_start + "result["+str(i) + "]/np.sum(result,axis=1),"
        cols = cols + str(i) +","
    cols = cols +"],"
    code_end=" baseline='zero')"
    code_all = code_start +cols + code_end
    f, axes = plt.subplots(1, 1, sharey=False, figsize=(8, 6))
    eval(code_all)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xlabel(xlabel='Quantile Of '+ features)
    axes.set_ylabel(ylabel='Percentage Of Differented ' + label)
    # 显示范围
    plt.xlim(0, 1)
    plt.ylim(ylim_min,ylim_max)
    # 添加图例
    plt.legend(loc='upper left')
    plt.grid(axis='y', color='gray', linestyle=':', linewidth=2)
    plt.title('Stacking Diagram Of ' + features)
    if save_fig:
        if not os.path.exists('./pictures/Stacking Diagram'):
            os.makedirs('./pictures/Stacking Diagram')
        plt.savefig('./pictures/Stacking Diagram/'+features+'.png',dpi=150)
        plt.close()


def stacking_diagram_compare(train, test, feature, label='label', decimals=2, ylim_min=0, ylim_max=1, save_fig=False):
    """
    对比两个数据集，趋势是否相同（可能随时间推移，特征发生变化），其他与stacking_diagram类似。
    :param train: 数据集1
    :param test: 数据集2
    :param feature: 特征（连续型）
    :param label:
    :param decimals: 1代表以10为间隔的分位点(可以看大致趋势)
    :param ylim_min:
    :param ylim_max:
    :param save_fig:
    :return:
    """
    train = train.copy()
    test = test.copy()
    #
    train[feature] = train[feature].fillna(-999)
    test[feature] = test[feature].fillna(-999)
    train[feature] =np.around(train[feature].rank(method='max',pct=True),decimals)
    test[feature] = np.around(test[feature].rank(method='max', pct=True), decimals)
    #
    result1 = train.groupby([feature,label]).size().unstack().fillna(1)
    result2 = test.groupby([feature,label]).size().unstack().fillna(1)
    #
    start_result1 = result1.head(1)
    start_result2 = result2.head(1)
    #
    start_result1.index=[0.0,]
    start_result2.index=[0.0,]
    #
    result1 = pd.concat([start_result1,result1],axis=0)
    result2 = pd.concat([start_result2,result2],axis=0)
    #
    code_start1 ="plt.stackplot(result1.index,"
    code_start2 = "plt.stackplot(result2.index,"
    #
    cols ="labels = ["
    label_unique = list(train[label].unique())
    label_unique.sort()
    for i in label_unique:
        code_start1 = code_start1 + "result1["+str(i) + "]/np.sum(result1,axis=1),"
        code_start2 = code_start2 + "result2[" + str(i) + "]/np.sum(result2,axis=1),"
        cols = cols + str(i) +","
    #cols = cols + "],colors=['#0485d1','#40a368','#fcb001','#f10c45'],"
    cols = cols +"], "
    code_end=" baseline='zero')"
    #str of plt code
    code_all1 = code_start1 +cols + code_end
    code_all2 = code_start2 +cols + code_end
    #
    f, axes = plt.subplots(1, 2, sharey=False, figsize=(16, 6))
    #Train stackplot
    plt.sca(axes[0])
    eval(code_all1)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].set_xlabel(xlabel='Quantile Of Train '+ feature)
    axes[0].set_ylabel(ylabel='Percentage Of Differented ' + label)
    # Test stackplot
    plt.sca(axes[1])
    eval(code_all2)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_xlabel(xlabel='Quantile Of Test '+ feature)
    axes[1].set_ylabel(ylabel='Percentage Of Differented ' + label)
    # 显示范围
    axes[0].set_xlim(0,1)
    axes[0].set_ylim(ylim_min,ylim_max)
    axes[1].set_xlim(0,1)
    axes[1].set_ylim(ylim_min,ylim_max)
    # 添加图例
    axes[0].legend(loc='lower right')
    axes[0].grid(axis='y', color='gray', linestyle=':', linewidth=2)
    #axes[0].title('Stacking Diagram Compare Of ' + feature)
    axes[1].legend(loc='lower right')
    axes[1].grid(axis='y', color='gray', linestyle=':', linewidth=2)
    #axes[1].title('Stacking Diagram Compare Of ' + feature)
    if save_fig:
        if not os.path.exists('./pictures/Stacking Diagram Compare'):
            os.makedirs('./pictures/Stacking Diagram Compare')
        plt.savefig('./pictures/Stacking Diagram Compare/'+feature+'.png',dpi=150)
        plt.close()
