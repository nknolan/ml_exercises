import tensorflow as tf

#Annotated with the tensorflow library definitions
#Easier to follow, hopefully
#This is a backpropagation neural network

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/mnist_data/", one_hot=True)

num_nodes_hidden1 = 500
num_nodes_hidden2 = 500
batch_size = 100
num_classes = 10
#placeholder() sets tensors for the input layer. The feed_dict in train_network() is necessary for these.
x = tf.placeholder('float', [None, 748])
y = tf.placeholder('float')

def network_model(data):
	#Define layers
	#tf.random_normal() produces a random array over a normal distribution
	hidden1 = {'weights':tf.Variable(tf.random_normal([748, num_nodes_hidden1])), 'biases':tf.Variable(tf.random_normal([num_nodes_hidden1]))}
	hidden2 = {'weights':tf.Variable(tf.random_normal([num_nodes_hidden1, num_nodes_hidden2])), 'biases':tf.Variable(tf.random_normal([num_nodes_hidden2]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([num_nodes_hidden2, num_classes])), 'biases':tf.Variable(tf.random_normal([num_classes]))}

	#Multiply data by weight, add biases
	#add() adds, matmul() performs matrix multiplication
	l1 = tf.add(tf.matmul(data, hidden1['weights']), hidden1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden2['weights']), hidden2['biases'])
	l2 = tf.nn.relu(l2)

	output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])
	return output

def train_network(x):
	prediction = network_model(x)
	#softmax_cross_entropy_with_logits is the cost function comparing the prediction to y.
	#logit maps the probabilities [0,1] to [-infinity, infinity]
	#Softmax maps [-inf, inf] to [0,1], like sigmoid(), but normalizes the sum of the output (a vector) to 1
	#This basically just uses softmax to normalize the data. tensorflow.org says its going to be deprecated soon
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	#AdamOptimizer() is apparently an implementation of stochastic gradient descent so that's what I'm using.
	#Learning rate is currently set to the default but I want it there just to visualize.
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
	num_epochs = 10
	
	#This is just a bunch of variables that took a while to figure out.
	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		for epochs in range(num_epochs):
			epoch_cost = 0
			for i in range(int(mnist.train.num_examples/batch_size)):
				#Breaks the mnist data set into nice chunks for the program to analyze.
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				i, c = session.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
				epoch_cost += c
				print("Epoch ", epoch, " complete out of ", num_epochs, ". Cost: ", epoch_cost)
		
		#tf.argmax() returns the index of the highest value. Since these are one-hot arrays, that's the index
		#of the prediction and the label. Then tf.equal() compares them.
		correct = equaltf.equal(tf.argmax(preditction, 1), tf.argmax(y, 1))
		#Identical to np.mean(). cast() converts correct from a tensor to a float.
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		
		print("Accuracy: ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
		
train_network(x)
