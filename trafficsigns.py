# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
import processor
from tensorflow.contrib.layers import flatten

training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

### To start off let's do a basic data summary.

n_train = len(X_train)

n_test = len(X_test)

x_dimension = len(X_train[0])
y_dimension = len(X_train[0][0])

n_classes = np.amax(y_train) + 1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", x_dimension, y_dimension)
print("Number of classes =", n_classes)

# Define your architecture here.
# Feel free to use as many code cells as needed.


# architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the architecture and return the result of the last fully connected layer.
def architecture(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    x = tf.reshape(x, (-1, 32, 32, 3))

    # C1: 32x32x16
    stride_c1 = 1
    filter_dimension_c1 = 3
    filter_depth_c1 = 16
    F_W_c1 = tf.Variable(tf.truncated_normal((filter_dimension_c1, filter_dimension_c1, 3, filter_depth_c1)))
    F_b_c1 = tf.Variable(tf.zeros(filter_depth_c1))
    strides_c1 = [1, stride_c1, stride_c1, 1]
    padding_c1 = 'SAME'

    x = tf.nn.conv2d(x, F_W_c1, strides_c1, padding_c1) + F_b_c1

    print("SHAPE AFTER C1={}", x.get_shape())

    # A1:
    x = tf.nn.relu(x)

    # MP1: 16x16x16
    strides_mp1 = [1, 2, 2, 1]
    filter_dimension_mp1 = 2
    padding_mp1 = 'VALID'

    x = tf.nn.max_pool(x, [1, filter_dimension_mp1, filter_dimension_mp1, 1], strides_mp1, padding_mp1)

    print("SHAPE AFTER MP1={}", x.get_shape())

    # C2: 12x12x128
    stride_c2 = 1
    filter_dimension_c2 = 5
    filter_depth_c2 = 128
    F_W_c2 = tf.Variable(
        tf.truncated_normal((filter_dimension_c2, filter_dimension_c2, filter_depth_c1, filter_depth_c2)))
    F_b_c2 = tf.Variable(tf.zeros(filter_depth_c2))
    strides_c2 = [1, stride_c2, stride_c2, 1]
    padding_c2 = 'VALID'

    x = tf.nn.conv2d(x, F_W_c2, strides_c2, padding_c2) + F_b_c2

    print("SHAPE AFTER C2={}", x.get_shape())

    # A2:
    x = tf.nn.relu(x)

    # MP2: 6x6x128
    strides_mp2 = [1, 2, 2, 1]
    filter_dimension_mp2 = 2
    padding_mp2 = 'VALID'

    x = tf.nn.max_pool(x, [1, filter_dimension_mp2, filter_dimension_mp2, 1], strides_mp2, padding_mp2)

    print("SHAPE AFTER MP2={}", x.get_shape())

    # FLATTEN
    x = flatten(x)

    # FC1: 4608->1000
    F_W_fc1 = tf.Variable(tf.random_normal((x.get_shape().as_list()[-1], 1000)))
    F_b_fc1 = tf.Variable(tf.zeros(1000))
    x = tf.add(tf.matmul(x, F_W_fc1), F_b_fc1)

    # A3:
    x = tf.nn.relu(x)

    # FC2: 1000->400
    F_W_fc2 = tf.Variable(tf.random_normal((1000, 400)))
    F_b_fc2 = tf.Variable(tf.zeros(400))
    x = tf.add(tf.matmul(x, F_W_fc2), F_b_fc2)

    # A4:
    x = tf.nn.relu(x)

    # FC3: 400->43
    F_W_fc3 = tf.Variable(tf.random_normal((400, 43)))
    F_b_fc3 = tf.Variable(tf.zeros(43))
    x = tf.add(tf.matmul(x, F_W_fc3), F_b_fc3)

    # Return the result of the last fully connected layer.
    return x


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.current_batch_index = 0

    def next_batch(self, batch_size):
        batch_X, batch_y = self.X[self.current_batch_index:self.current_batch_index + batch_size], \
                           self.y[self.current_batch_index:self.current_batch_index + batch_size]
        self.current_batch_index += batch_size
        return batch_X, batch_y

    def reset(self):
        self.current_batch_index = 0


def eval_data(dataset, batch_size):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    steps_per_epoch = len(dataset.X) // batch_size
    num_examples = steps_per_epoch * batch_size
    total_acc, total_loss = 0, 0

    for step in range(steps_per_epoch):
        batch_X, batch_y = dataset.next_batch(batch_size)

        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_X, y: batch_y})
        total_acc += (acc * batch_X.shape[0])
        total_loss += (loss * batch_X.shape[0])

    dataset.reset()

    return total_loss / num_examples, total_acc / num_examples

# NOTE: Feel free to change these.
EPOCHS = 100
BATCH_SIZE = 64


processed_data = processor.process_data(train, test)

train_dataset = Dataset(processed_data['train']['X'], processed_data['train']['y'])
validate_dataset = Dataset(processed_data['validate']['X'], processed_data['validate']['y'])
test_dataset = Dataset(processed_data['test']['X'], processed_data['test']['y'])

# Dataset consists of 32x32x1, grayscale images.
flatten_input = processed_data['train']['X'].shape[1]
flatten_output = processed_data['train']['y'].shape[1]

x = tf.placeholder(tf.float32, (None, flatten_input))

# Classify over number of classes (43).
y = tf.placeholder(tf.float32, (None, flatten_output))
# Create the architecture.
fc2 = architecture(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps_per_epoch = len(train_dataset.X) // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE

    # Train model
    for i in range(EPOCHS):
        for step in range(steps_per_epoch):
            batch_X, batch_y = train_dataset.next_batch(BATCH_SIZE)
            loss = sess.run(train_op, feed_dict={x: batch_X, y: batch_y})

        train_dataset.reset()

        val_loss, val_acc = eval_data(validate_dataset, BATCH_SIZE)
        print("EPOCH {} ...".format(i + 1))
        print("Validation loss = {}".format(val_loss))
        print("Validation accuracy = {}".format(val_acc))

    # Evaluate on the test data
    test_loss, test_acc = eval_data(test_dataset, BATCH_SIZE)
    print("Test loss = {}".format(test_loss))
    print("Test accuracy = {}".format(test_acc))