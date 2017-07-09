import mxnet as mx
import logging

mnist = mx.test_utils.get_mnist()
batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

data = mx.sym.var('data')

data = mx.sym.flatten(data=data)

# Multi layer perceptron with rectified linear unit as activation function
fc1  = mx.sym.FullyConnected(data = data, num_hidden = 128)
act1 = mx.sym.Activation(data = fc1, act_type = "relu")

fc2 = mx.sym.FullyConnected(data = data, num_hidden = 64)
act2 = mx.sym.Activation(data = fc2, act_type = "relu")


# MNIST has 10 classes

fc3 = mx.sym.FullyConnected(data = act2, num_hidden = 10)

# Software with cross entropy loss or back propogation
mlp = mx.sym.SoftmaxOutput(data = fc3, name = 'softmax')

logging.getLogger().setLevel(logging.DEBUG)

# Create trainable module on CPU

mlp_model = mx.mod.Module(symbol = mlp, context = mx.cpu())
mlp_model.fit(train_iter, optimizer = "sgd", optimizer_params= {'learning_rate': 0.1},
        eval_metric = 'acc', batch_end_callback = mx.callback.Speedometer(batch_size, 100), num_epoch = 10)

test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = mlp_model.predict(test_iter)

assert prob.shape == (10000, 10)

test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.96
