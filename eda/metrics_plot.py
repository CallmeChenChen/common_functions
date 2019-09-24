import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve, auc

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#计算ks值
def compute_ks(prob, target):
    '''
    target: numpy array of shape (1,)
    proba: numpy array of shape (1,), predicted probability of the sample being positive
    returns:
    ks: float, ks score estimation
    '''
    get_ks = lambda prob, target: ks_2samp(prob[target == 1], prob[target != 1]).statistic

    return get_ks(prob, target)

#学习曲线
def plot_learning(estimator, X_train, y_train, cv=10):
    '''
    estimator: 模型
    X_train: array n-D
    y_train: array 1-D
    '''
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    train_size, train_score , test_score = learning_curve(
        estimator=estimator, X=X_train, y=y_train, cv=cv, n_jobs=-1,train_sizes= np.linspace(0.1,1,10)
    )
    train_mean = np.mean(train_score, axis=1)
    train_std = np.std(train_score, axis=1)
    test_mean = np.mean(test_score, axis=1)
    test_std = np.std(test_score, axis=1)

    plt.plot(train_size, train_mean, color ='blue', marker='o',markersize=5,
             label = 'training acc')
    plt.fill_between(train_size, train_mean + train_std,
             train_mean - train_std,
             alpha = 0.15, color = 'blue')
    plt.plot(train_size, test_mean,color='green',linestyle='--',
             marker='s', markersize=5,label='validation acc')

    plt.fill_between(train_size, test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('training samples')
    plt.ylabel('Acc')
    plt.legend(loc='lower right')
    # plt.ylim([0.8,1.0])
    plt.show()

#验证曲线
def plot_validation(estimator, X_train, y_train, param_name, param_range, cv=10):
    train_score, test_score = validation_curve(
        estimator=estimator, X=X_train, y=y_train,
        param_name=param_name,param_range=param_range,cv=cv
    )

    train_mean = np.mean(train_score,axis=1)
    train_std = np.std(train_score,axis=1)
    test_mean = np.mean(test_score,axis=1)
    test_std = np.std(test_score,axis=1)

    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5,
             label = 'training acc')
    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha = 0.15, color='blue')

    plt.plot(param_range, test_mean,color='green', linestyle='--',
             marker='s',markersize=5, label='validation acc')
    plt.fill_between(param_range, test_mean + test_std,
                     test_mean - test_std, alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('参数{}'.format(param_name))
    plt.ylabel('Acc')
    plt.show()

#AUC ROC

def plot_roc(prob, target):
    '''
    画 ROC
    :param target: 0和1目标值
    :param proba: 概率值或分数
    :return:
    '''
    #返回 假正率，真正率，阈值
    fpr,tpr,thresholds = roc_curve(target, prob, pos_label=0)
    roc_auc = auc(fpr,tpr)
    print("AUC Value:{}".format(roc_auc))
    lw = 2
    plt.figure(figsize=(8,6))
    #横坐标为假正率，纵坐标为真正率
    plt.plot(fpr, tpr, color='darkorange',
         lw=1.2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.grid(color='g',linestyle='-.')
    plt.show()

#
def plot_ks(prob, target):
    '''
    画ks
    :param target: 0和1目标值
    :param proba: 概率值或分数
    :return:
    '''
    plt.figure(figsize=(8,6))
    a = pd.DataFrame(np.array([prob, target]).T, columns=['proba', 'target'])
    a.sort_values(by='proba', ascending=True, inplace=True)
    a['sum_Times'] = a['target'].cumsum()
    total_1 = a['target'].sum()
    total_0 = len(a) - a['target'].sum()

    a['temp'] = 1
    a['Times'] = a['temp'].cumsum()
    a['cdf1'] = a['sum_Times'] / total_1
    a['cdf0'] = (a['Times'] - a['sum_Times']) / total_0
    a['ks'] = a['cdf1'] - a['cdf0']
    a['percent'] = a['Times'] * 1.0 / len(a)

    idx = np.argmax(a['ks'])
    print('KS value:{}'.format(a.loc[idx]['ks']))
    #计算x轴
    proba_min = min(a['proba'])
    proba_max = max(a['proba'])
    a['x'] = a['proba'].apply(lambda x:(x-proba_min)/(proba_max - proba_min))
    plt.plot(a['x'], a['cdf1'], label="CDF_positive")
    plt.plot(a['x'], a['cdf0'], label="CDF_negative")
    plt.plot(a['x'], a['ks'], label="K-S")
    plt.legend()
    plt.grid(True)
    ymin, ymax = plt.ylim()
    #plt.xlabel('Sample percent')
    plt.ylabel('Cumulative probability')
    plt.title('Model Evaluation Index K-S')
    plt.axis('tight')

    # 虚线
    t = a.loc[idx]['x']
    yb = round(a.loc[idx]['cdf1'], 4)
    yg = round(a.loc[idx]['cdf0'], 4)
    plt.plot([t, t], [yb, yg], color='red', linewidth=1.4, linestyle="--")
    plt.scatter([t, ], [yb, ], 20, color='dodgerblue')

    # K-S曲线峰值
    plt.scatter([t, ], [a.loc[idx]['ks'], ], 20, color='limegreen')
    plt.annotate(r'ks=%s' % (round(a.loc[idx]['ks'], 4))
                 , xy=(a.loc[idx]['x'], a.loc[idx]['ks'])
                 , xycoords='data'
                 , xytext=(+15, -15),
                 textcoords='offset points'
                 , fontsize=13
                 , arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))
    plt.show()


#评分违约率曲线
def plot_score_default(test_score_df, bins=10):
    """
    :param test_score_df: 包含 score 和 target 变量
    :param bins:
    :return:
    """
    test_score_df['category'] = pd.qcut(test_score_df['score'], bins)
    col_names = {'mean': 'default', 'count_nonzero': 'bad_ops'}
    grouped = test_score_df.groupby('category')
    summary = grouped.agg([np.mean, np.size, np.count_nonzero])['y'].rename(columns=col_names)
    summary['good_ops'] = summary['size'] - summary['bad_ops']
    summary['category'] = summary.index
    print(summary)

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    ax1.bar(range(len(summary)), summary['bad_ops'], label='bad', color='#3c9992', align='center')
    ax1.bar(range(len(summary)), summary['good_ops'], bottom=summary['bad_ops'], label='good', color='#f1f33f', align='center')
    ax1.set_ylabel(u'频数')
    ax1.set_title(u'信用评分模型稳定性效果图')
    ax1.set_ylim([0, max(summary['size']) + 16])
    ax1.set_xlabel(u'信用评分区间')
    ax1.legend(loc='upper right')
    plt.xticks([0,1,2,3,4,5,6,7,8,9],summary['category'],rotation=20)
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(range(len(summary)), summary['default'], 'r', marker='o', mec='r', mfc='w', label=u'逾期率', lw=1.2, ls='-')
    ax2.set_xlim([-1, 10])
    ax2.set_ylim([0, 1])
    ax2.set_ylabel(u'逾期率')

    default = list(map(lambda x:str(round(x, 2)) + '%', summary['default']))
    for i, (x, y) in enumerate(zip(range(len(summary)), summary['default'])):
        plt.text(x + 0.02, y + 0.02, default[i], color='red', fontsize=10)
    plt.show()


def plot_ks_roc(prob, target, out_path=False):
    fig = plt.figure(figsize=(8,6))
    fig.add_subplot(121)
    fpr, tpr, thresholds = roc_curve(target, prob, pos_label=0)
    roc_auc = auc(fpr, tpr)
    print("AUC Value:{}".format(roc_auc))
    plt.plot(fpr, tpr, color='darkorange',
             lw=1.2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.grid()

    fig.add_subplot(122)
    a = pd.DataFrame(np.array([prob, target]).T, columns=['proba', 'target'])
    a.sort_values(by='proba', ascending=True, inplace=True)
    a['sum_Times'] = a['target'].cumsum()
    total_1 = a['target'].sum()
    total_0 = len(a) - a['target'].sum()

    a['temp'] = 1
    a['Times'] = a['temp'].cumsum()
    a['cdf1'] = a['sum_Times'] / total_1
    a['cdf0'] = (a['Times'] - a['sum_Times']) / total_0
    a['ks'] = a['cdf1'] - a['cdf0']
    a['percent'] = a['Times'] * 1.0 / len(a)

    idx = np.argmax(a['ks'])
    print('KS value:{}'.format(a.loc[idx]['ks']))
    #计算x轴
    proba_min = min(a['proba'])
    proba_max = max(a['proba'])
    a['x'] = a['proba'].apply(lambda x:(x-proba_min)/(proba_max - proba_min))
    plt.plot(a['x'], a['cdf1'], label="CDF_positive")
    plt.plot(a['x'], a['cdf0'], label="CDF_negative")
    plt.plot(a['x'], a['ks'], label="K-S")
    plt.legend()
    plt.grid(True)
    ymin, ymax = plt.ylim()
    #plt.xlabel('Sample percent')
    plt.ylabel('Cumulative probability')
    plt.title('Model Evaluation Index K-S')
    plt.axis('tight')

    # 虚线
    t = a.loc[idx]['x']
    yb = round(a.loc[idx]['cdf1'], 4)
    yg = round(a.loc[idx]['cdf0'], 4)
    plt.plot([t, t], [yb, yg], color='red', linewidth=1.4, linestyle="--")
    plt.scatter([t, ], [yb, ], 20, color='dodgerblue')

    # K-S曲线峰值
    plt.scatter([t, ], [a.loc[idx]['ks'], ], 20, color='limegreen')
    plt.annotate(r'ks=%s,p=%s$' % (round(a.loc[idx]['ks'], 4)
                                    , round(a.loc[idx]['proba'], 4))
                 , xy=(a.loc[idx]['x'], a.loc[idx]['ks'])
                 , xycoords='data'
                 , xytext=(+15, -15),
                 textcoords='offset points'
                 , fontsize=10
                 , arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))
    if out_path:
        plt.savefig('{}.jpg'.format(out_path))