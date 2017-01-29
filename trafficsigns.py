# Load pickled data
import pickle
import numpy as np
import cv2
import random
import math
import tensorflow as tf

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


### Preprocess the data here.
### Feel free to use as many code cells as needed.

def transform_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_n_img_per_class(labels):
    n_img_per_class = []
    current_y = 0
    current_count = 0

    for y in labels:
        if y == current_y:
            current_count += 1
        else:
            current_y = y
            n_img_per_class.append(current_count)
            current_count = 1

    n_img_per_class.append(current_count)
    return n_img_per_class


def extract_validation_set(X_train, y_train):
    total_images, rows, cols = X_train.shape

    new_X_train = np.copy(X_train)
    new_y_train = np.copy(y_train)

    X_validate = np.empty((0, rows, cols), dtype=X_train.dtype)
    y_validate = np.array([], dtype=y_train.dtype)

    n_img_per_class = get_n_img_per_class(y_train)
    start_index = 0

    for n_img in n_img_per_class:
        n_picks = int(n_img / 10)

        index_interval = list(range(start_index, start_index + n_img))
        index_list = np.random.choice(index_interval, n_picks, replace=False)
        index_list = np.sort(index_list)

        X_validate = np.append(X_validate, np.take(X_train, index_list, 0), 0)
        y_validate = np.append(y_validate, np.take(y_train, index_list))

        new_X_train = np.delete(new_X_train, index_list, 0)
        new_y_train = np.delete(new_y_train, index_list)

        start_index = start_index + n_img

    return {
        'X_train': new_X_train,
        'y_train': new_y_train,
        'X_validate': X_validate,
        'y_validate': y_validate
    }


def rotate_image(img, angle):
    rows, cols = img.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def translate_image(img, translation_x, translation_y):
    rows, cols = img.shape

    M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def scale_image(img, scale):
    rows, cols = img.shape
    res = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    new_rows, new_cols = res.shape

    if (scale > 1):
        res = res[int(new_rows / 2) - 16:int(new_rows / 2) + 16, int(new_cols / 2) - 16:int(new_cols / 2) + 16]
    else:
        res = cv2.copyMakeBorder(res, math.ceil((32 - new_rows) / 2), int((32 - new_rows) / 2),
                                 math.ceil((32 - new_cols) / 2), int((32 - new_cols) / 2), cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])

    return res


def add_jitter(image):
    jitter_image = scale_image(image, random.uniform(0.9, 1.1))
    jitter_image = rotate_image(jitter_image, random.uniform(-15.0, 15.0))
    jitter_image = translate_image(jitter_image, random.randrange(-2, 2), random.randrange(-2, 2))
    return jitter_image


# Taken from https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/preprocessing.py
def contrast_normalize(X, scale=1., subtract_mean=True, use_std=True, sqrt_bias=10., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).
    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.
    scale : float, optional
        Multiply features by this const.
    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.
    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.
    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.
    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.
    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.
    Notes
    -----
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].
    References
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # First, local contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    X = clahe.apply(X)

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, np.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X


def prepare_data(X):
    # Preliminary: transform all images into YUV format, only take Y
    X_grey = np.array(list(map(transform_grey, X)))

    # Finally, we do global and local contrast normalization for the images
    X_norm = np.array(list(map(contrast_normalize, X_grey)))

    return X_norm


# -------------- DATA PREP PIPELINE

# First, we do global and local contrast normalization for the images

X_train_prep = prepare_data(X_train)
X_test_prep = prepare_data(X_test)

print("train and test data has been pre-processed")

# Then, validation set selected at random per class
# Validation set extraction:
validate_set = extract_validation_set(X_train_prep, y_train)

X_train_remaining = validate_set['X_train']
y_train_remaining = validate_set['y_train']

X_validate = validate_set['X_validate']
y_validate = validate_set['y_validate']

print("validation data has been selected")

# After we have the validation set, we generate jitter images of the remanining training set:
# Duplicate the X_train list by adding jitter
X_jitter = np.array(list(map(add_jitter, X_train_remaining)))
y_jitter = y_train_remaining

X_train_and_jitter = np.append(X_train_remaining, X_jitter, 0)
y_train_and_jitter = np.append(y_train_remaining, y_jitter)

print("added jitter data to dataset")

### Define your architecture here.
### Feel free to use as many code cells as needed.

"""
architecture Architecture

HINTS for layers:

    Convolutional layers:

    tf.nn.conv2d
    tf.nn.max_pool

    For preparing the convolutional layer output for the
    fully connected layers.

    tf.contrib.flatten
"""
from tensorflow.contrib.layers import flatten

# NOTE: Feel free to change these.
EPOCHS = 100
BATCH_SIZE = 64


# architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the architecture and return the result of the last fully connected layer.
def architecture(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    x = tf.reshape(x, (-1, 32, 32, 1))

    # C1: 32x32x16
    stride_c1 = 1
    filter_dimension_c1 = 3
    filter_depth_c1 = 16
    F_W_c1 = tf.Variable(tf.truncated_normal((filter_dimension_c1, filter_dimension_c1, 1, filter_depth_c1)))
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
        batch_X, batch_y = self.X[self.current_batch_index:self.current_batch_index + batch_size], self.y[
                                                                                                   self.current_batch_index:self.current_batch_index + batch_size]
        self.current_batch_index += batch_size
        return batch_X, batch_y

    def reset(self):
        self.current_batch_index = 0


def eval_data(dataset):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    steps_per_epoch = len(dataset.X) // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0

    for step in range(steps_per_epoch):
        batch_X, batch_y = dataset.next_batch(BATCH_SIZE)

        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_X, y: batch_y})
        total_acc += (acc * batch_X.shape[0])
        total_loss += (loss * batch_X.shape[0])

    dataset.reset()

    return total_loss / num_examples, total_acc / num_examples


def one_hot(y, n_labels):
    input_length = len(y)

    one_hot_encoded = np.zeros((input_length, n_labels))
    one_hot_encoded[np.arange(input_length), y] = 1
    return one_hot_encoded


# Load data
def flatten_dataset(X):
    n_examples, rows, columns = X.shape
    return np.reshape(X, (-1, rows * columns))


train_dataset = Dataset(flatten_dataset(X_train_and_jitter), one_hot(y_train_and_jitter, n_classes))
validate_dataset = Dataset(flatten_dataset(X_validate), one_hot(y_validate, n_classes))
test_dataset = Dataset(flatten_dataset(X_test_prep), one_hot(y_test, n_classes))

# Dataset consists of 32x32x1, grayscale images.
x = tf.placeholder(tf.float32, (None, 1024))
# Classify over number of classes (43).
y = tf.placeholder(tf.float32, (None, n_classes))
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

        val_loss, val_acc = eval_data(validate_dataset)
        print("EPOCH {} ...".format(i + 1))
        print("Validation loss = {}".format(val_loss))
        print("Validation accuracy = {}".format(val_acc))

    # Evaluate on the test data
    test_loss, test_acc = eval_data(test_dataset)
    print("Test loss = {}".format(test_loss))
    print("Test accuracy = {}".format(test_acc))