# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

class Ridge_:
    def __init__(self, alpha=1, copy_X=True, normalize=True,
                 solver='gd', solve_dJ='debug', n_iters=1000, learning_rate=0.1, tol=1e-3):
        # n_iters、 learning_rate、 tol在选用梯度下降和随机梯度下降法时有效， solve_dJ仅在选择梯度下降法时有效
        assert solver in ['gd', 'sgd'], \
            'solver must be one of "gd", "sgd", and "{}" got!'.format(solver)
        assert solve_dJ in ['debug'], \
            'solve_dJ must be one of "debug", and "{}" got!'.format(solve_dJ)
        self.alpha = alpha
        self.copy_X = copy_X
        self.normalize = normalize
        self.solver = solver
        self.solve_dJ = solve_dJ
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.tol = tol
        
    def _fit_lsq(self, X_b, y):  # 最小二乘法训练模型
        XT_X = X_b.T.dot(X_b) + self.alpha * np.eye(X_b.shape[1])
        self._theta = np.linalg.inv(XT_X).dot(X_b.T.dot(y))  # (X^T * X + alpha*I)^-1 * (X^T * y)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

class Lasso_:
    def __init__(self, alpha=1, copy_X=True, normalize=True,
                 solver='gd', solve_dJ='debug', n_iters=1000, learning_rate=0.1, tol=1e-3):
        # n_iters、 learning_rate、 tol在选用梯度下降和随机梯度下降法时有效， solve_dJ仅在选择梯度下降法时有效
        assert solver in ['gd', 'sgd'], \
            'solver must be one of "gd", "sgd", and "{}" got!'.format(solver)
        assert solve_dJ in ['debug'], \
            'solve_dJ must be one of "debug", and "{}" got!'.format(solve_dJ)
        self.alpha = alpha
        self.copy_X = copy_X
        self.normalize = normalize
        self.solver = solver
        self.solve_dJ = solve_dJ
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.tol = tol

    def _fit_gd(self, X_b, y):  # 梯度下降法训练模型
        theta = np.ones(X_b.shape[1])
        for _ in range(self.n_iters):
            dJ = _dJ_gd_debug(theta, X_b, y, J='Lasso', alpha=self.alpha)
            theta -= self.learning_rate * dJ
            if np.linalg.norm(dJ) <= self.tol:  # 提前终止条件
                break
        self._theta = theta
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

def main(data):
    x,y = read_data()
    weight1 = Ridge_(x,y)
    weight2 = Lasso_(x,y)
    return weight1,weight2

def read_data(path='C:/Users/EthanCai/linear-regression-ccyyqq981217/data/exp02'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
