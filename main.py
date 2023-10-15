from algorithms import feedforward_neural_networks
from tqdm import tqdm
import numpy as np
from keras.datasets import mnist
from keras.src.utils import to_categorical


# 设置参数
num_epochs = 10
batch_size = 600
num_batches = int(60000 / batch_size)

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 将图像数据转换为模型可用的格式，并进行归一化
train_images = train_images.reshape((train_images.shape[0], 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28 * 28)).astype('float32') / 255

# 对标签进行 one-hot 编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 打乱数据
indices = np.arange(train_images.shape[0])
np.random.shuffle(indices)
train_images = train_images[indices]
train_labels = train_labels[indices]


# 创建前馈神经网络模型
model = feedforward_neural_networks()
# 将 MNIST 数据传递给模型进行训练
model.train()
last_loss = 200
for epoch in range(num_epochs):
	sum_loss = 0.0
	progress_bar = tqdm(range(num_batches), desc=f'Epoch [{epoch + 1}/{num_epochs}]')
	for i in progress_bar:
		batch_images = train_images[i * batch_size: (i + 1) * batch_size]
		batch_labels = train_labels[i * batch_size: (i + 1) * batch_size]
		# 反向传播
		batch_loss = model.backward(batch_images, batch_labels)
		sum_loss += batch_loss.item() / 60000  # 总共60000个训练样本
		progress_bar.set_postfix(loss=sum_loss)
	if sum_loss > last_loss:
		break
	last_loss = sum_loss

# 将模型设置为评估模式
model.eval()
# 在测试集上进行预测
predictions = model.forward(test_images)
# 获取预测结果中每个样本最大概率的索引，作为模型的预测标签
predicted_labels = np.argmax(predictions, axis=1)
# 将真实标签中每个样本最大概率的索引，作为真实标签
true_labels = np.argmax(test_labels, axis=1)
# 计算准确率
accuracy = np.mean(predicted_labels == true_labels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
