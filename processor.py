import cv2
import numpy as np
import math
import random

### Preprocess the data here.
### Feel free to use as many code cells as needed.


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
    total_images, rows, cols, color_depth = X_train.shape

    new_X_train = np.copy(X_train)
    new_y_train = np.copy(y_train)

    X_validate = np.empty((0, rows, cols, color_depth), dtype=X_train.dtype)
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

# Taken from https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/preprocessing.py
def _contrast_normalize(X, scale=1., subtract_mean=True, use_std=True, sqrt_bias=10., min_divisor=1e-8):
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

def _rotate_image(img, angle):
    rows = img.shape[0]
    cols = img.shape[1]

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def _translate_image(img, translation_x, translation_y):
    rows = img.shape[0]
    cols = img.shape[1]

    M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def _scale_image(img, scale):
    res = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    new_rows = res.shape[0]
    new_cols = res.shape[1]

    if (scale > 1):
        res = res[int(new_rows / 2) - 16:int(new_rows / 2) + 16, int(new_cols / 2) - 16:int(new_cols / 2) + 16]
    else:
        res = cv2.copyMakeBorder(res, math.ceil((32 - new_rows) / 2), int((32 - new_rows) / 2),
                                 math.ceil((32 - new_cols) / 2), int((32 - new_cols) / 2), cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])

    return res

def _add_jitter(image):
    jitter_image = _scale_image(image, random.uniform(0.9, 1.1))
    jitter_image = _rotate_image(jitter_image, random.uniform(-15.0, 15.0))
    jitter_image = _translate_image(jitter_image, random.randrange(-2, 2), random.randrange(-2, 2))
    return jitter_image


def _preprocess_image(X):
    # Preliminary: transform all images into YUV format, only take Y
    X_grey = np.array(list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), X)))

    # Finally, we do global and local contrast normalization for the images
    X_norm = np.array(list(map(_contrast_normalize, X_grey)))

    return X_norm


# Load data
def _flatten_dataset(X):
    shape = X.shape
    n_features = 1

    for i in range(1, len(shape)):
        n_features *= shape[i]

    return np.reshape(X, (-1, n_features))


def _one_hot(y, n_labels):
    input_length = len(y)

    one_hot_encoded = np.zeros((input_length, n_labels))
    one_hot_encoded[np.arange(input_length), y] = 1
    return one_hot_encoded


def process_data(train, test):
    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    n_classes = np.amax(y_train) + 1

    # -------------- DATA PREP PIPELINE

    # First, we do global and local contrast normalization for the images

    # X_train = preprocess_image(X_train)
    # X_test = preprocess_image(X_test)

    print("train and test data has been pre-processed")

    # Then, validation set selected at random per class
    # Validation set extraction:
    validate = extract_validation_set(X_train, y_train)

    X_train_remaining = validate['X_train']
    y_train_remaining = validate['y_train']

    X_validate = validate['X_validate']
    y_validate = validate['y_validate']

    print("validation data has been selected")

    # After we have the validation set, we generate jitter images of the remanining training set:
    # Duplicate the X_train list by adding jitter
    # TODO: check if compensating for amount of features per label helps

    X_jitter = np.array(list(map(_add_jitter, X_train_remaining)))
    y_jitter = y_train_remaining

    X_train_with_jitter = np.append(X_train_remaining, X_jitter, 0)
    y_train_with_jitter = np.append(y_train_remaining, y_jitter)

    print("added jitter data to dataset")

    # Flatten features
    X_train_with_jitter = _flatten_dataset(X_train_with_jitter)
    X_validate = _flatten_dataset(X_validate)
    X_test = _flatten_dataset(X_test)

    # One-hot encode labels
    y_train_with_jitter = _one_hot(y_train_with_jitter, n_classes)
    y_validate = _one_hot(y_validate, n_classes)
    y_test = _one_hot(y_test, n_classes)

    return {
        'train': {
            'X': X_train_with_jitter,
            'y': y_train_with_jitter
        },
        'validate': {
            'X': X_validate,
            'y': y_validate
        },
        'test': {
            'X': X_test,
            'y': y_test
        }
    }