import pandas as pd
import numpy as np
import random
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
import os
import tensorflow as tf
import cv2
import time
from time import gmtime, strftime
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
# Load pickled data
training_file = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

rate = 0.002
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
np.unique(y_train, return_counts=True)


SEED = 12345
np.random.seed(SEED)
random.seed(SEED)
index = random.randint(0, len(X_train))


tf.reset_default_graph()
KEEP_PROB = 0.7
from tensorflow.contrib.layers import flatten


def rgb_to_gray(rgb):
    x_shape = list(rgb.shape)
    x_shape[3] = 1
    gray = np.dot(rgb, [0.2989, 0.5870, 0.1140])
    return gray.reshape(x_shape)

def imgs_to_gray(X):
    x_shape = list(X.shape)
    x_shape[3] = 1
    return np.mean(X, axis=3).reshape(x_shape)

def rgb_to_hsv_n(imgs):
    return np.array([rgb_to_hsv(imgs[i].squeeze()) for i in range(len(imgs))])

def normalize_img(img):
    return (img - np.min(img))/(np.max(img) - np.min(img))
    
def normalize_imgs(imgs):
    return np.array([normalize_img(imgs[i]) for i in range(len(imgs))])
if X_train.max() > 2:
#     X_train = imgs_to_gray(X_train)
#     X_valid = imgs_to_gray(X_valid)
#     X_test = imgs_to_gray(X_test)
#     X_train = (X_train - 128.)/128.
#     X_valid = (X_valid - 128.)(/128.
#     X_test = (X_test - 128.)/128.
    
    X_train = X_train/255.
    X_valid = X_valid/255.
    X_test = X_test/255.
    
    X_train = rgb_to_gray(X_train)
    X_valid = rgb_to_gray(X_valid)
    X_test = rgb_to_gray(X_test)
    
#     X_train = normalize_imgs(X_train)
#     X_valid = normalize_imgs(X_valid)
#     X_test = normalize_imgs(X_test)
    
#     X_train = rgb_to_hsv_n(X_train)
#     X_valid = rgb_to_hsv_n(X_valid)
#     X_test = rgb_to_hsv_n(X_test)

#     X_train = np.concatenate((rgb_to_gray(X_train), rgb_to_hsv_n(X_train)[:,:,:,:2]), axis=3)
#     X_valid = np.concatenate((rgb_to_gray(X_valid), rgb_to_hsv_n(X_valid)[:,:,:,:2]), axis=3)
#     X_test = np.concatenate((rgb_to_gray(X_test), rgb_to_hsv_n(X_test)[:,:,:,:2]), axis=3)
else:
    print("Already normalized")


n_color = X_train.shape[-1]


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return x

def relu(x):
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='VALID')

def avgpool2d(x, k=2):
    return tf.nn.avg_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='VALID')

# Weights Initializer
mu = 0
sigma = 0.1
    
w1 = tf.Variable(tf.random_normal([5, 5, n_color, 8], mean=mu, stddev=sigma, seed=SEED))
b1 = tf.Variable(tf.random_normal([8], mean=mu, stddev=sigma, seed=SEED))

w2 = tf.Variable(tf.random_normal([5, 5, 8, 20], mean=mu, stddev=sigma, seed=SEED))
b2 = tf.Variable(tf.random_normal([20], mean=mu, stddev=sigma, seed=SEED))

w3 = tf.Variable(tf.random_normal([500, 160], mean=mu, stddev=sigma, seed=SEED))
b3 = tf.Variable(tf.random_normal([160], mean=mu, stddev=sigma, seed=SEED))    

w4 = tf.Variable(tf.random_normal([160, 120], mean=mu, stddev=sigma, seed=SEED))
b4 = tf.Variable(tf.random_normal([120], mean=mu, stddev=sigma, seed=SEED))

w5 = tf.Variable(tf.random_normal([120, n_classes], mean=mu, stddev=sigma, seed=SEED))
b5 = tf.Variable(tf.random_normal([n_classes], mean=mu, stddev=sigma, seed=SEED))

def LeNet(x, keep_prob=KEEP_PROB):
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x8.
    conv1 = conv2d(x, w1, b1, strides=1)
    
    # Activation.
    conv1 = relu(conv1)
    
    # dropout
    conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob, seed=SEED)

    # Pooling. Input = 28x28x8. Output = 14x14x8.
#     pool1 = avgpool2d(conv1, k=2)
    pool1 = maxpool2d(conv1, k=2)

    # Layer 2: Convolutional. Output = 10x10x20.
    conv2 = conv2d(pool1, w2, b2, strides=1)
    
    # Activation.
    conv2 = relu(conv2)
    
    # dropout
    conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob, seed=SEED)

    # Pooling. Input = 10x10x20. Output = 5x5x20.
#     pool2 = avgpool2d(conv2, k=2)
    pool2 = maxpool2d(conv2, k=2)
    
    # Flatten. Input = 5x5x20. Output = 500.
    flat = tf.reshape(pool2, [-1, 500])
    
    # Layer 3: Fully Connected. Input = 500. Output = 160.
    fully_connected_1 = tf.add(tf.matmul(flat, w3), b3)
    
    # Activation.
    fully_connected_1 = relu(fully_connected_1)
    
    # dropout
    fully_connected_1 = tf.nn.dropout(fully_connected_1, keep_prob=keep_prob, seed=SEED)
    
    # Layer 4: Fully Connected. Input = 160. Output = 120.
    fully_connected_2 = tf.add(tf.matmul(fully_connected_1, w4), b4)
    
    # Activation.
    fully_connected_2 = relu(fully_connected_2)
    
    # dropout
    fully_connected_2 = tf.nn.dropout(fully_connected_2, keep_prob=keep_prob, seed=SEED)

    # Layer 5: Fully Connected. Input = 120. Output = n_classes.
    logits = tf.add(tf.matmul(fully_connected_2, w5), b5)
    
    return logits
x = tf.placeholder(tf.float32, (None, 32, 32, n_color))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)

one_hot_y = tf.one_hot(y, n_classes)

BATCH_SIZE = 1
#model_name = './traffic_sign_classifier_{}_{}_{}_{}'.format(strftime("%Y%m%d_%H%M%S", gmtime()), BATCH_SIZE, rate, KEEP_PROB)
model_name = './traffic_sign_classifier_20191125_010558_512_0.002_0.7'

logits = LeNet(x, keep_prob)
def predict(X_data):
    num_examples = len(X_data)
    sess = tf.get_default_session()
    y_list = []
    file_handle = open("4.txt", mode='w')

    for offset in range(0, num_examples, BATCH_SIZE):
        import time
        batch_x = X_data[offset:offset+BATCH_SIZE]
        start_time = time.time()
        batch_y = sess.run(tf.argmax(logits, 1), feed_dict={x: batch_x, keep_prob: 1.})
        print(batch_y)
        file_handle.write(str(time.time()-start_time)+'\n')
        y_list.append(batch_y)
    file_handle.close()
    return np.hstack(y_list)

def predict_prob(X_data):
    num_examples = len(X_data)
    sess = tf.get_default_session()
    prob_list = []
    prob_ids_list = []
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x = X_data[offset:offset+BATCH_SIZE]
        batch_prob, batch_prob_ids = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k=5),
                                              feed_dict={x: batch_x, keep_prob: 1.})
        prob_list.append(batch_prob)
        prob_ids_list.append(batch_prob_ids)
    return np.hstack(prob_list), np.hstack(prob_ids_list)




df_web_images = pd.read_csv("web_image_labels_nvprof.csv")  ###nvprof
X_web_test = np.array([cv2.resize(cv2.imread(img_i), (32,32)) for img_i in df_web_images['image_name'].tolist()])
if X_web_test.shape[-1]==3:
    X_web_test = X_web_test/255.
    X_web_test = rgb_to_gray(X_web_test)
else:
    print("Already converted to gray")
model_meta = tf.train.import_meta_graph("{}.meta".format(model_name))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model_meta.restore(sess, './traffic_sign_classifier_20191125_010558_512_0.002_0.7')
#     y_web_pred = predict(X_web_test)
    import time
    y_web_pred = predict(X_web_test)
    print(y_web_pred)
    y_pred_prob, y_pred_ids = predict_prob(X_web_test)
#     y_web_pred = y_pred_ids[:, 0]

df_labels = pd.read_csv("signnames.csv", index_col=0)
df_labels.head()
df_web_images['predicted_label'] = y_web_pred
df_web_images['predicted_label_name'] = df_labels['SignName'].iloc[y_web_pred].values
## Calculate the accuracy for these test images from the web. 
web_images_acc = (df_web_images['label_id'] == df_web_images['predicted_label']).sum()/len(df_web_images)*100
print("Accuracy for test images from the web: {:.1f}%".format(web_images_acc))

