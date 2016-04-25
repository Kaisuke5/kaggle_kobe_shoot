import numpy as np
import six
import chainer.links as L
import chainer
print chainer.__version__
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers


class network(chainer.Chain):

	def __init__(self, n_in, n_units, n_out):
		super(network, self).__init__(
			l1=L.Linear(n_in, n_units),
			l2=L.Linear(n_units, n_units),
			l3=L.Linear(n_units, n_out),
		)

	def __call__(self, x, y):
		t = self.predict(x)
		self.loss =  F.mean_squared_error(y, t)
		return self.loss

	def predict(self, x):
		h1 = F.relu(self.l1(x))
		h2 = F.relu(self.l2(h1))
		t = self.l3(h2)
		return t


class shoot_network():

	def __init__(self,units=300,gpu=-1):
		self.units = units
		self.model = None
		self.gpu = gpu

		self.gpu = gpu
		if self.gpu >= 0:
			from chainer import cuda
			self.xp = cuda.cupy
			cuda.get_device(self.gpu).use()
		else:
			self.xp = self.xp




	def fit(self,x_train,y_train,n_epoch=100, batchsize=30):
		x_train = self.xp.array(x_train, self.xp.float32)
		y_train = self.xp.array(y_train, self.xp.float32).reshape(len(y_train),1)
		self.train(x_train, y_train, n_epoch=n_epoch, batchsize=30)

	def train(self, x_train, y_train,n_epoch=100, batchsize=30):
		self.model = network(len(x_train[0]),self.units,1)
		optimizer = optimizers.Adam()
		optimizer.setup(self.model)
		N = len(x_train)
		for epoch in six.moves.range(1, n_epoch + 1):
			print('epoch', epoch)
			# training
			perm = self.xp.random.permutation(N)
			sum_accuracy = 0
			sum_loss = 0
			for i in six.moves.range(0, N, batchsize):
				x = chainer.Variable(self.xp.asarray(x_train[perm[i:i + batchsize]]))
				t = chainer.Variable(self.xp.asarray(y_train[perm[i:i + batchsize]]))
				# Pass the loss function (Classifier defines it) and its arguments
				optimizer.update(self.model, x, t)

				sum_loss += float(self.model.loss.data) / N

			print 'epoch %d mean_squared_error:%f' % (epoch,sum_loss)


	def predict(self,test_x):
		x = chainer.Variable(self.xp.asarray(test_x, self.xp.float32))
		t = self.model.predict(x)
		return t.data







