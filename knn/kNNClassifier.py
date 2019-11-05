import numpy as np
from collections import Counter
from math import sqrt
from model_selection.metrics import accuracy_score

class kNNClassifier:

    def __init__(self, k):
        '''初始化kNN分类器'''
        assert k>=1
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0]
        assert self.k<= X_train.shape[0]

        self._X_train = X_train
        self._y_train = y_train
        return self # 返回自身，与scikit-learn保持一致，以便日后对接

    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None
        assert X_predict.shape[1] == self._X_train.shape[1] #特征维数相同

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict) # 返回多个预测结果

    def _predict(self, x):
        # 私有的predict
        # 计算距离
        distance = [sqrt(np.sum((x_ - x) ** 2)) for x_ in self._X_train]
        # 距离排序
        nearest = np.argsort(distance)
        # 找出前k个值
        topk_y = [self._y_train[i] for i in nearest[:self.k]]
        # 统计投票数
        votes = Counter(topk_y)
        # 返回投票最高的
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        #根据xtest与ytest判断模型准确度

        y_predict = self.predict(X_test)
        return accuracy_score(y_test,y_predict)

    def __repr__(self):
        # 打印k
        return "KNN(k=%d)" % self.k

