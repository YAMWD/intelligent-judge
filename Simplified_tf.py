import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.preprocessing  as sp
import pandas as pd
import random
import math
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

def load():
	index = 7
	csv_data = pd.read_csv("data.csv", skiprows = [])
	list = csv_data.values.tolist()
	train_X = []
	train_Y = []
	for i in range(len(list)):
		x = list[i][2:18]
		y = list[i][18:19]
		if(y[0] == 0):
			y[0] = 0.1
		if(x[index] <= 30000):
			train_X.append(x)
			train_Y.append(y)

	train_X = np.array(train_X) 
	train_Y = np.array(train_Y)

	return train_X, train_Y

def load_data(i, num_val_samples, data, target):
	index = 7

	test_X = data[i * num_val_samples : (i + 1) * num_val_samples]
	test_Y = target[i * num_val_samples : (i + 1) * num_val_samples]

	train_X = np.concatenate([data[: i * num_val_samples], data[(i + 1) * num_val_samples:]], axis = 0)
	train_Y = np.concatenate([target[: i * num_val_samples], target[(i + 1) * num_val_samples:]], axis = 0)

	#归一化
	Max = train_X.max(axis = 0)
	Min = train_X.min(axis = 0)
	for i in range(train_X.shape[0]):
		train_X[i][index] = (train_X[i][index] - Min[index]) / (Max[index] - Min[index])
	for i in range(test_X.shape[0]):
		test_X[i][index] = (test_X[i][index] - Min[index]) / (Max[index] - Min[index])

	#标准化
	'''mean = train_X.mean(axis = 0)
	std = train_X.std(axis = 0)
	train_X = (train_X - mean) / std
	test_X = (test_X - mean) / std'''

	'''with open('record.txt', 'w') as f:
		f.write(str(Max))
		f.write('\n')
		f.write(str(Min))'''


	return train_X, train_Y, test_X, test_Y

def Load_data():
	index = 7
	csv_data = pd.read_csv("data.csv", skiprows = [])
	list = csv_data.values.tolist()
	train_X = []
	train_Y = []
	for i in range(len(list)):
		x = list[i][2:18]
		y = list[i][18:19]
		if(y[0] == 0):
			y[0] = 0.1
		if(x[index] <= 30000 and y[0] < 10):
			train_X.append(x)
			train_Y.append(y)

	train_X = np.array(train_X) 
	train_Y = np.array(train_Y)

	train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size = 0.1, random_state = 8)

	money_stolen1 = []
	for i in range(train_Y.shape[0]):
		money_stolen1.append(train_X[i][index])

	money_stolen2 = []
	for i in range(test_Y.shape[0]):
		money_stolen2.append(test_X[i][index])

	plt.subplot(2, 1, 1)
	plt.hist(money_stolen1, bins = train_X.shape[0], rwidth = 0.8)
	plt.title('Train')
	plt.ylabel('Number')
	plt.legend()

	plt.subplot(2, 1, 2)
	plt.hist(money_stolen2, bins = test_X.shape[0], rwidth = 0.8)
	plt.title('Test')
	plt.xlabel('Money_Stolen')
	plt.ylabel('Number')
	plt.legend()

	plt.show()

	#归一化
	Max = train_X.max(axis = 0)
	Min = train_X.min(axis = 0)
	for i in range(train_X.shape[0]):
		train_X[i][index] = (train_X[i][index] - Min[index]) / (Max[index] - Min[index])
	for i in range(test_X.shape[0]):
		test_X[i][index] = (test_X[i][index] - Min[index]) / (Max[index] - Min[index])

	#标准化
	'''mean = train_X.mean(axis = 0)
	std = train_X.std(axis = 0)
	train_X = (train_X - mean) / std
	test_X = (test_X - mean) / std'''

	'''with open('record.txt', 'w') as f:
		f.write(str(Max))
		f.write('\n')
		f.write(str(Min))'''

	return train_X, train_Y, test_X, test_Y

def Shuffle_data(train_X, train_Y, test_X, test_Y):
	New_Train_X = []
	New_Train_Y = []
	Len = train_X.shape[0]
	Sequence = np.random.permutation(Len)
	for i in Sequence:
		New_Train_X.append(train_X[i])
		New_Train_Y.append(train_Y[i])

	New_Train_X = np.array(New_Train_X)
	New_Train_Y = np.array(New_Train_Y)

	New_Test_X = []
	New_Test_Y = []
	Len = test_X.shape[0]
	Sequence = np.random.permutation(Len)
	for i in Sequence:
		New_Test_X.append(test_X[i])
		New_Test_Y.append(test_Y[i])

	New_Test_X = np.array(New_Test_X)
	New_Test_Y = np.array(New_Test_Y)

	return New_Train_X, New_Train_Y, New_Test_X, New_Test_Y

def get_weight(shape, lambda1, LayerName):
	#获取初始化权重并加入L2正则项
	var = tf.get_variable(str(LayerName) + 'weights', shape, initializer = tf.contrib.layers.xavier_initializer())
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
	return var

def add_layer(inputs, input_size, output_size, lambda1, LayerName, activation_function = None):
	with tf.name_scope(LayerName):
		with tf.name_scope('Weights'):
			Weights = get_weight([input_size, output_size], lambda1, LayerName)
			tf.summary.histogram(LayerName + '/Weights', Weights)

		with tf.name_scope('Bias'):
			bias = tf.Variable(tf.zeros([1, output_size]) + 0.1, name = 'B')
			tf.summary.histogram(LayerName + '/Bias', bias)

		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs, Weights) + bias
			Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)

		tf.summary.histogram(LayerName + '/Output', outputs)
		return outputs

def build_network(x, layers, lambda1):
	z = x
	Len = len(layers)
	for i in range(Len - 2):
		z = add_layer(z, layers[i], layers[i + 1], lambda1, 'hidden_layer' + str(i), activation_function = tf.nn.elu)

	y_pred = add_layer(z, layers[Len - 2], layers[Len - 1], lambda1, 'output_layer', activation_function = None)

	return y_pred
	
def delete_logs():

	path = './logs/train'

	files = os.listdir(path)

	for file in files:
		c_path = os.path.join(path, file)
		os.remove(c_path)

	path = './logs/test'

	files = os.listdir(path)
	for file in files:
		c_path = os.path.join(path, file)
		os.remove(c_path)


#神经网络结构
delete_logs()#删除tensorboard文件，避免overwriting
data, target = load()
file_path = "Model/Small_Test/model"
input_size = data.shape[1]
output_size = target.shape[1]
batch_size = data.shape[0]
lambda1 = 0.03
learning_rate = 0.001
kk = 1
num_val_samples = data.shape[0] // kk
all_scores = []
all_acc3 = []
all_acc6 = []

for fold in range(kk):
	train_X, train_Y, test_X, test_Y = Load_data()
	#train_X, train_Y, test_X, test_Y = load_data(fold, num_val_samples, data, target)
	input_size = train_X.shape[1]
	output_size = train_Y.shape[1]
	batch_size = train_X.shape[0]

	tf.reset_default_graph()
	#预留每个batch中输入与输出的空间
	with tf.name_scope('Inputs'):
		x = tf.placeholder(tf.float32, shape = (None, input_size), name = 'X_input')
		y = tf.placeholder(tf.float32, shape = (None, output_size), name = 'Y_input')
		keep_prob = tf.placeholder(tf.float32)

	layer = [input_size, 64, 32, output_size]

	y_pred = build_network(x, layer, lambda1)

	#MSE
	with tf.name_scope('MSE'):
		loss = (tf.reduce_mean(tf.square(y_pred - y))) / 2
		train_loss = (tf.reduce_mean(tf.square(y_pred - y))) / 2
		tf.summary.scalar('train_loss', train_loss)
		test_loss = (tf.reduce_mean(tf.square(y_pred - y))) / 2
		tf.summary.scalar('test_loss', test_loss)

	#MAE
	with tf.name_scope('MAE'):
		Loss = (tf.reduce_mean(tf.abs(y_pred - y)))
		tf.summary.scalar('Loss', loss)

	'''L2正则化'''
	tf.add_to_collection('losses', loss)
	L2_loss = tf.add_n(tf.get_collection('losses'))

	with tf.name_scope('Accuracy'):
		train_acc = 1 - tf.reduce_mean(tf.abs(y_pred - y) / 5)
		tf.summary.scalar('train_acc', train_acc)
		test_acc = 1 - tf.reduce_mean(tf.abs(y_pred - y) / 5)
		tf.summary.scalar('test_acc', test_acc)

	with tf.name_scope('Train'):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(L2_loss)#反向传播优化

	'''with tf.name_scope('Train'):
		train_step = tf.train.MomentumOptimizer(learning_rate = 0.01, momentum = 0.9).minimize(loss)#SGD'''

	'''with tf.name_scope('Train'):
		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)#反向传播优化'''
	#创建会话
	saver = tf.train.Saver()
	with tf.Session() as sess:

		merged = tf.summary.merge_all()

		train_writer = tf.summary.FileWriter('logs/train', sess.graph)

		test_writer = tf.summary.FileWriter('logs/test', sess.graph)

		init_op = tf.global_variables_initializer()

		sess.run(init_op)#初始化变量

		#saver.restore(sess, file_path)

		#锁定图
		tf.Graph().finalize()

		#设定训练次数
		epoch = 1000000
		k = 0
		for k in range(epoch):
			#k += 1

			#设置每一次训练样本个数
			a = 0
			b = train_X.shape[0]

			#Shuffle
			train_X, train_Y, test_X, test_Y = Shuffle_data(train_X, train_Y, test_X, test_Y)

			a = 0
			while True:
				b = min(a + batch_size, train_X.shape[0])
				#calculate
				sess.run(train_step, feed_dict = {x:train_X[a:b], y:train_Y[a:b], keep_prob:1})
				a = b
				if(a == train_X.shape[0]):
					break

			#显示误差
			if k % 100 == 0:

				total_loss1 = sess.run(Loss, feed_dict = {x:train_X, y:train_Y, keep_prob:1})

				val_loss1 = sess.run(Loss, feed_dict = {x:test_X, y:test_Y, keep_prob:1})

				total_acc1 = sess.run(train_acc, feed_dict = {x:train_X, y:train_Y, keep_prob:1})

				val_acc1 = sess.run(test_acc, feed_dict = {x:test_X, y:test_Y, keep_prob:1})

				train_result = sess.run(merged, feed_dict = {x:train_X, y:train_Y, keep_prob:1})

				val_result = sess.run(merged, feed_dict = {x:test_X, y:test_Y, keep_prob:1})

				train_writer.add_summary(train_result, k)

				test_writer.add_summary(val_result, k)

				train_writer.flush()

				test_writer.flush()

				predict_train = sess.run(y_pred, feed_dict = {x:train_X, keep_prob:1})

				predict_test = sess.run(y_pred, feed_dict = {x:test_X, keep_prob:1})

				print("Processing fold %d，训练%d个epoch后，loss为%f，Acc为%f，val_loss为%f，val_Acc为%f" % (fold, k, total_loss1, total_acc1, val_loss1, val_acc1))

		save_path = saver.save(sess, file_path)

		all_scores.append(val_loss1)

		predict = sess.run(y_pred, feed_dict = {x:test_X, keep_prob:1})

		acc3 = 0
		acc6 = 0

		for i in range(test_X.shape[0]):
			if(abs(predict[i][0] - test_Y[i][0] <= 0.25)):
				acc3 += 1
			if(abs(predict[i][0] - test_Y[i][0] <= 0.5)):
				acc6 += 1

		all_acc3.append(acc3 / test_X.shape[0])
		all_acc6.append(acc6 / test_X.shape[0])

		print(acc3 / test_X.shape[0])
		print(acc6 / test_X.shape[0])

		'''index = train_X.shape[0]
		width = 0.3
		plt.figure()
		plt.grid(True)

		predict = sess.run(y_pred, feed_dict = {x:train_X, keep_prob:1})
		accuracy = []
		p = []
		for i in range(train_Y.shape[0]):
			accuracy.append(predict[i][0])
			p.append(train_Y[i][0])

		plt.subplot(2, 1, 1)
		plt.bar(range(train_Y.shape[0]), accuracy, width, color = 'blue', label = "Simulation")
		plt.bar(np.arange(train_Y.shape[0]) + width, p, width, color = 'orange', label = "Fact")
		plt.title('Train')
		plt.ylabel('Sentence')
		plt.legend()

		predict = sess.run(y_pred, feed_dict = {x:test_X, keep_prob:1})
		accuracy = []
		p = []
		for i in range(test_Y.shape[0]):
			accuracy.append(predict[i][0])
			p.append(test_Y[i][0])
		plt.subplot(2, 1, 2)
		plt.bar(range(test_Y.shape[0]), accuracy, width, color = 'blue', label = "Simulation")
		plt.bar(np.arange(test_Y.shape[0]) + width, p, width, color = 'orange', label = "Fact")
		plt.title('Test')
		plt.xlabel('Number')
		plt.ylabel('Sentence')
		plt.legend()

		plt.show()'''

print(all_scores)
print(np.mean(all_scores))
print(all_acc3)
print(np.mean(all_acc3))
print(all_acc6)
print(np.mean(all_acc6))
