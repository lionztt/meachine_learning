import numpy as np
from collections import Counter
from math import sqrt

def knn_classify(k,x_train,y_train,x):
    #加入一些断言保证输入合法
    assert 1<=k<=x_train.shape[0]
    assert x_train.shape[0] == y_train.shape[0]
    assert x_train.shape[1] == x.shape[0]

    # 计算距离
    distance = [sqrt(np.sum((x_ - x) ** 2)) for x_ in x_train]
    # 距离排序
    nearest = np.argsort(distance)
    # # 设定k值
    # k = 6
    # 找出前k个值
    topk_y = [y_train[i] for i in nearest[:k]]
    # 统计投票数
    votes = Counter(topk_y)
    # 返回投票最高的
    return votes.most_common(1)[0][0]

