import numpy as np
class Perceptron:
    # 梯度下降法
    def __init__(self, lr = 0.01, epochs = 1000):
        self.lr = lr
        self.epochs = epochs
        # 初始化参数 步长和学习次数
    def fit(self, X, y):
        #训练函数
        n, d = X.shape
        # X为训练集，n为训练集的个数，d为每个样本的维度
        self.w = np.random.randn(d) # 参数随机方法初始化 d维向量
        self.b = np.random.randn(1) # 随机数初始化偏移量
        # 初始化，w是感知机对应的线性权重矩阵，b是偏移度
        for _ in range(self.epochs):
            margins = y * (X @ self.w + self.b)
            # 按照当前的参数w和b所计算出来的结果
            # margins中小于 0 的点意味着分类错误
            mis_idx = margins <= 0
            # 寻找分类错误的点
            if not np.any(mis_idx):
                break
                #如果没有点分类错误就退出
            X_mis = X[mis_idx]
            y_mis = y[mis_idx]
            # 提取所有错误分类的点的数据点
            # 梯度下降法
            self.w += self.lr * (y_mis[:, np.newaxis] * X_mis).sum(axis = 0)
            # 这里是把一维向量转化为二维数组，从而可以和X做乘法，最后按照行进行相加
            self.b += self.lr * y_mis.sum()
            # 一次性处理 这里是利用公式 self.lr是步长 y.mis.sum()是累计错误的点，分类错误一次就要加上一次
    def predict(self, X):
        return np.sign(X @ self.w + self.b)
        # 一个函数 返回值就是
class Perceptron_Random:
    # 随机梯度下降法
    def __init__(self, lr = 0.01, epochs = 1000):
        # lr和epochs是训练参数，但是有初始化,类比c++中的构造函数
        self.lr = lr
        self.epochs = epochs
    def fit(self, X, y):
        n, d = X.shape
        # n是训练集数量，d是维度
        self.w = np.zeros(d)
        # w和b是模型参数
        self.b = 0.0
        # w和b视作是Preceptron自身的属性 将会运用到预测函数当中
        for _ in range(self.epochs):
            # 执行次数是self.epochs
            margin = y * (X @ self.w + self.b)
            mis_idx = margin <= 0
            if not np.any(mis_idx):
                break
            i = np.random.randint(n)
            # 随机找一条
            if y[i] * (X[i] @ self.w + self.b) <= 0:
                self.w += self.lr * y[i] * X[i]
                self.b += self.lr * y[i]
        # 如果判断是分类错误就加一条
    def predict(self, X):
        # 模型给出的预测结果
        return np.sign(X @ self.w + self.b)
class NonlinearPerceptron_XOR:
    def __init__(self, lr = 0.1, epochs = 10000):
        self.lr = lr
        self.epochs = epochs
    def ReLU(self, X):
        return np.maximum(0, X) # 返回ReLU函数的值
    def Sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x)) # 返回矩阵或者数字的Sigmoid值
    def Derivative_ReLU(self, X):
        return (X > 0).astype(float)
    def Derivative_Sigmoid(self, x):
        return self.Sigmoid(x) * (1 - self.Sigmoid(x)) # ReLU函数的梯度
    def fit(self, X, y):
        n, d = X.shape
        # 获取维度和数据集数量
        hidden_dim = 6  # 隐藏层的个数
        np.random.seed(110) # 设置随机数种子
        self.w1 = np.random.randn(d, hidden_dim) * 0.1
        # 对self.w1, 它的每一列是某一个数据集的参数
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros((1, 1)) # 初始化 这里都用矩阵的方式来表示
        # 这里是初始化 用随机数初始化
        y = y.reshape(-1, 1)
        # 确保y是二维矩阵
        for epochs in range(self.epochs):
            # 循环
            # 先计算损失误差
            z1 = X @ self.w1 + self.b1
            h1 = self.ReLU(z1) # 第一层 隐藏层用ReLU
            z2 = h1 @ self.w2 + self.b2
            h2 = self.Sigmoid(z2) # 第二层 输出层用Sigmoid 更适合二分类问题
            error = h2 - y # 计算误差 损失函数
            delta2 = error * self.Derivative_Sigmoid(z2)
            dw2 = h1.T @ delta2 / n
            db2 = np.sum(delta2, axis = 0, keepdims = True) / n
            delta1 = (delta2 @ self.w2.T) * self.Derivative_ReLU(z1)
            dw1 = X.T @ delta1 / n
            db1 = np.sum(delta1, axis = 0, keepdims = True) / n
            self.w2 -= self.lr * dw2
            self.b2 -= self.lr * db2
            self.w1 -= self.lr * dw1
            self.b1 -= self.lr * db1
            if epochs % 1000 == 0:
                print(f"Epoch {epochs}, Loss: {np.mean(error ** 2):.6f}")  # ← 加这行

            # 随机生成一个参数
    def predict(self, x):
        z1 = x @ self.w1 + self.b1
        h1 = self.ReLU(z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = self.Sigmoid(z2)
        return (h2 > 0.5).astype(int)[0][0]
def randomData(n, d, w, b):
    # 随机生成n条d维数组
    # 随机生成直线
    X = np.random.uniform(-10, 10, (n, d))
    # X 是 -10 到 10 之间的数据
    y = (X @ w + b >= 0).astype(int) * 2 - 1
    # 把布尔类型转化为感知机的分类类型
    return X, y
def Perceptron_Train_Test():
    # 感知机
    print(" =========== ")
    dim = 2    # 数据维度
    train_data_num = 1000    # 训练样本数量
    test_data_num = 100    # 测试样本数量
    w = np.random.uniform(-1, 1, dim)    # 理论生成的w值
    b = np.random.uniform(-1, 1)    # 理论生成的b值
    X_train ,y_train = randomData(train_data_num, dim, w, b)    # 测试集
    X_test, y_test = randomData(test_data_num, dim, w, b)    # 训练集
    model = Perceptron(lr = 0.1, epochs = 1000000)    # 用感知机算法
    model.fit(X_train, y_train)    # 训练模型
    y_pred = model.predict(X_test)    # 测试模型
    accuracy = np.mean(y_pred == y_test) # 准确率
    print(f" Perceptron : {accuracy:.3f}")
    model_rand = Perceptron_Random(lr = 0.1, epochs = 100000)
    model.fit(X_train, y_train) # 随机梯度下降法模型训练
    y_pred_rand = model.predict(X_test) # 模型效果查看
    accuracy_rand = np.mean(y_pred_rand == y_test) # 随机梯度下降法
    print(f" Perceptron_Random : {accuracy_rand:.3f}") # 准确率
def randomData_XOR(n = 4, d = 2):
    # 二维异或问题数据生成
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # 这里得到X.shape = (4, 2)
    # 每一行都都是一个数据
    #
    y = np.array([1, 0, 0, 1])  # 参数
    return X, y
def XOR_Train_Test():
    print(" =========== ")
    print(" XOR ")
    X,y = randomData_XOR() # 获取随机数据
    model = NonlinearPerceptron_XOR() # 获取模型
    model.fit(X, y)     # 训练模型
    for i in range(2):
        for j in range(2):
            print(f" {i} xor {j} : {model.predict([i, j])}")
def main():
    # 主函数
    Perceptron_Train_Test()
    # 基础感知机算法
    XOR_Train_Test()
    # 二维异或问题 使用激活函数和梯度下降法
if __name__ == "__main__":
    main()
