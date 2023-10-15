import numpy as np


def sigmoid(x, gard=False):
	z = np.exp(x)
	h = z / (1 + z)
	if gard:
		return h * (1 - h)
	return h


def tanh(x, gard=False):
	h = np.tanh(x)
	if gard:
		return 1 - h * h
	return h


def relu(x, gard=False, leaky=0):
	if gard:
		return np.piecewise(x, [x > 0, x <= 0], [lambda x: 1, lambda x: leaky]) + 0
	return np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: leaky * x]) + 0


def softmax(x):
	exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
	return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class linear:
	def __init__(self, input_size, output_size, learning_rate):
		# 初始化权重和偏置
		self.weights = np.random.randn(input_size, output_size)
		self.bias = np.zeros((1, output_size))
		# 用于存储前向传播的输入和输出
		self.input = None
		self.output = None
		# 用于存储梯度
		self.grad_weights = None
		self.grad_bias = None
		# 训练模式
		self.train_mode = True
		# 学习率
		self.learning_rate = learning_rate

	def train(self):
		self.train_mode = True

	def eval(self):
		self.train_mode = False

	def forward(self, input):
		if self.train_mode:
			# 保存输入，计算输出
			self.input = input
			self.output = np.dot(input, self.weights) + self.bias
			return self.output
		return np.dot(input, self.weights) + self.bias

	def backward(self, grad_output):
		# 计算权重和偏置的梯度
		self.grad_weights = np.dot(self.input.T, grad_output) / self.input.shape[0]
		self.grad_bias = np.mean(grad_output, axis=0, keepdims=True)
	
		# 计算输入的梯度
		grad_input = np.dot(grad_output, self.weights.T)
		# 更新权重和偏置
		self.weights += self.learning_rate * self.grad_weights
		self.bias -= self.learning_rate * self.grad_bias
		return grad_input


def categorical_cross_entropy(targets, output):
	epsilon = 1e-15
	output = np.clip(output, epsilon, 1 - epsilon)
	loss = -np.sum(targets * np.log(output), axis=1)
	batch_loss = np.sum(loss)
	return batch_loss


def mean_squared_error(targets, predictions):
	differ = targets - predictions
	return np.sum(differ * differ) / targets.shape[0]


class feedforward_neural_networks:
	def __init__(self, learning_rate=0.1):
		self.linear1 = linear(784, 512, learning_rate)
		self.linear2 = linear(512, 256, learning_rate)
		self.linear3 = linear(256, 128, learning_rate)
		self.linear4 = linear(128, 64, learning_rate)
		self.linear5 = linear(64, 10, learning_rate)

	def train(self):
		self.linear1.train()
		self.linear2.train()
		self.linear3.train()
		self.linear4.train()
		self.linear5.train()

	def eval(self):
		self.linear1.eval()
		self.linear2.eval()
		self.linear3.eval()
		self.linear4.eval()
		self.linear5.eval()

	def forward(self, x):
		x_ = sigmoid(self.linear1.forward(x))
		x_ = sigmoid(self.linear2.forward(x_))
		x_ = sigmoid(self.linear3.forward(x_))
		x_ = sigmoid(self.linear4.forward(x_))
		return softmax(self.linear5.forward(x_))

	def backward(self, x, targets):
		# 计算损失函数和损失梯度, 通过网络反向传播梯度
		output = self.forward(x)
		batch_loss = categorical_cross_entropy(targets, output)
		loss_gradient = self.linear5.backward(targets - output)
		loss_gradient = self.linear4.backward(loss_gradient * sigmoid(self.linear4.output, True))
		loss_gradient = self.linear3.backward(loss_gradient * sigmoid(self.linear3.output, True))
		loss_gradient = self.linear2.backward(loss_gradient * sigmoid(self.linear2.output, True))
		self.linear1.backward(loss_gradient * sigmoid(self.linear1.output, True))
		return batch_loss
