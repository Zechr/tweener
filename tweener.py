from __future__ import division
import tensorflow as tf
import math
import numpy as np
import png
from scipy import misc
from queue import PriorityQueue
import random
import pflow

imagelist = []
with open('imagelist.txt', 'r') as f:
    imagelist = f.readlines()
imagelist = [im.strip() for im in imagelist]

WIDTH = 320
HEIGHT = 180

INPUT_DIM = WIDTH*HEIGHT
OUTPUT_DIM = INPUT_DIM
BATCH = 4
ITERS = 2048

current = 0

rerun = 0
shist = [[], [], [], []]

NSIZE = 3

w = png.Writer(WIDTH, HEIGHT, greyscale=True)
np.set_printoptions(threshold=np.nan)

def gray(img):
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def binary(img):
    img[img < 127] = 0
    img[img >= 127] = 1
    return img

def hex(img):
    return np.clip(0, np.multiply(img, 255), 255)

for i in range(0, len(imagelist), 3):
    imstart = np.array(misc.imread(imagelist[i]))
    imend = np.array(misc.imread(imagelist[i+2]))
    imbetween = np.array(misc.imread(imagelist[i+1]))
    
    imstart = gray(imstart)
    imend = gray(imend)
    imbetween = gray(imbetween)
    flow = pflow.optical_flow(imstart, imend, NSIZE)

    #with open(str(i/3+1) + '-flow.png', 'wb+') as f:
    #    w.write(f, np.reshape(pflow.flow_conversion(imstart, flow), (-1, WIDTH*1)))

    #shist[0].append(flow_conversion(np.subtract(imend, imstart), flow))
    #shist[0].append(np.concatenate((imstart[...,np.newaxis], imend[...,np.newaxis]), axis=2))
    shist[0].append(pflow.flow_conversion(imstart, flow))
    shist[1].append(imend)
    shist[2].append(imbetween)
    #shist[2].append(flow)
    shist[3].append(np.sum(np.square(np.subtract(imend, imstart))))

"""
for j in range(len(shist[0])):
    #resultimg = np.add(shist[2][j], shist[1][j])
    resultimg = hex(shist[2][j])
    resultimg = np.clip(resultimg, 0, 255).astype(int)
    with open(str(j+1) + '.png', 'wb+') as f:
        w.write(f, np.reshape(resultimg, (-1, WIDTH*1)))
"""

x_s = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
x_e = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
flow = tf.placeholder(tf.int32, shape=[None, HEIGHT, WIDTH])
org_loss = tf.placeholder(tf.float32, shape=[BATCH, 1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d(x, W, shape):
    return tf.nn.conv2d_transpose(x, W, output_shape=shape, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def next_batch(size):
    indices = random.sample(range(len(shist[0])), size)
    x_s = []
    x_e = []
    y = []
    org_l = []
    for index in indices:
        x_s.append(shist[0][index])
        x_e.append(shist[1][index])
        y.append(shist[2][index])
        org_l.append(shist[3][index])
    return {'x_s': np.array(x_s, dtype=object), 'x_e': np.array(x_e, dtype=object), 
    'y': np.array(y, dtype=object), 'org_l': np.array(org_l)}

W_conv0 = weight_variable([5, 5, 1, 8])
b_conv0 = bias_variable([8])
x_l = tf.reshape(x_s, [-1, HEIGHT, WIDTH, 1])
x_r = tf.reshape(x_e, [-1, HEIGHT, WIDTH, 1])
h_conv0_l = tf.nn.relu(conv2d(x_l, W_conv0) + b_conv0)
h_pool0_l = max_pool_2x2(h_conv0_l)
h_conv0_r = tf.nn.relu(conv2d(x_r, W_conv0) + b_conv0)
h_pool0_r = max_pool_2x2(h_conv0_r)

W_conv1 = weight_variable([3, 3, 8, 64])
b_conv1 = bias_variable([64])
h_conv1_l = tf.nn.relu(conv2d(h_pool0_l, W_conv1) + b_conv1)
h_pool1_l = max_pool_2x2(h_conv1_l)
h_conv1_r = tf.nn.relu(conv2d(h_pool0_r, W_conv1) + b_conv1)
h_pool1_r = max_pool_2x2(h_conv1_r)

W_conv2 = weight_variable([3, 3, 64, 128])
b_conv2 = bias_variable([128])
h_conv2_l = tf.nn.relu(conv2d(h_pool1_l, W_conv2) + b_conv2)
h_conv2_r = tf.nn.relu(conv2d(h_pool1_r, W_conv2) + b_conv2)


h_conv3 = h_conv2_l + h_conv2_r

W_deconv0 = weight_variable([3, 3, 64, 128])
b_deconv0 = bias_variable([64])
h_deconv0 = tf.nn.relu(deconv2d(h_conv3, W_deconv0, [tf.shape(x_l)[0], int(HEIGHT/4), int(WIDTH/4), 64]) + b_deconv0)

W_deconv1 = weight_variable([3, 3, 8, 64])
b_deconv1 = bias_variable([8])
h_deconv1 = tf.nn.relu(deconv2d(h_deconv0, W_deconv1, [tf.shape(x_l)[0], int(HEIGHT/4), int(WIDTH/4), 8]) + b_deconv1)
h_depool1 = tf.image.resize_images(h_deconv1, [int(HEIGHT/2), int(WIDTH/2)])

W_deconv2 = weight_variable([5, 5, 1, 8])
b_deconv2 = bias_variable([1])
h_deconv2 = tf.nn.relu(deconv2d(h_depool1, W_deconv2, [tf.shape(x_l)[0], int(HEIGHT/2), int(WIDTH/2), 1]) + b_deconv2)
h_depool2 = tf.image.resize_images(h_deconv2, [HEIGHT, WIDTH])

#W_deconv3 = weight_variable([3, 3, 1, 1])
#b_deconv3 = bias_variable([1])
#h_deconv3 = tf.nn.relu(deconv2d(h_depool2, W_deconv3, [tf.shape(x_features)[0], HEIGHT, WIDTH, 1]) + b_deconv3)

W_reconv0 = weight_variable([3, 3, 1, 1])
b_reconv0 = bias_variable([1])
y_conv = tf.reshape(tf.nn.relu(conv2d(h_depool2, W_reconv0) + b_reconv0), [-1, OUTPUT_DIM])

keep_prob = tf.placeholder(tf.float32)

loss = tf.reduce_mean(tf.divide(tf.reduce_sum(tf.square(tf.subtract(y_, y_conv)), axis=1), org_loss))


#Discriminator network
x = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
#binary classification
y_class = tf.placeholder(tf.float32, shape=[None, 2])

W_d_conv0 = weight_variable([5, 5, 1, 8])
b_d_conv0 = bias_variable([8])

h_d_conv0 = tf.nn.relu(conv2d(x, W_d_conv0) + b_d_conv0)
h_d_pool0 = max_pool_2x2(h_d_conv0)

W_d_conv1 = weight_variable([3, 3, 8, 16])
b_d_conv1 = bias_variable([16])

h_d_conv1 = tf.nn.relu(conv2d(h_d_pool0, W_d_conv1) + b_d_conv1)
h_d_pool1 = max_pool_2x2(h_d_conv1)

W_d_conv2 = weight_variable([3, 3, 16, 32])
b_d_conv2 = bias_variable([32])

h_d_conv2 = tf.nn.relu(conv2d(h_d_pool1, W_d_conv2) + b_d_conv2)

h_d_r = tf.reshape(h_d_conv2, [-1, WIDTH/4 * HEIGHT/4 * 32])
W_d_fc0 = weight_variable([WIDTH/4 * HEIGHT/4 * 32, 1024])
b_d_fc0 = bias_variable([1024])

h_d_fc0 = tf.nn.relu(tf.matmul(h_d_r, W_d_fc0) + b_d_fc0)

W_d_fc1 = weight_variable([1024, 2])
b_d_fc1 = bias_variable([2])

out = tf.matmul(h_d_fc0, W_d_fc1) + b_d_fc1

#discriminator loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_class, logits=out))
train_step_d = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#generator loss with discriminator loss included
loss_with_d = loss - 0.2*cross_entropy
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


sess = tf.InteractiveSession()
saver = tf.train.Saver()

if rerun == 0:
    sess.run(tf.global_variables_initializer())
if rerun == 1:
    saver.restore(sess, 'interpolation-model')
    print("Model restored")


accuracy = tf.reduce_mean(tf.divide(tf.reduce_sum(tf.square(tf.subtract(y_, y_conv)), axis=1), org_loss))

prediction = y_conv

for i in range(ITERS+1):
    batch = next_batch(BATCH)
    if i%BATCH == 0:
        train_accuracy = accuracy.eval(feed_dict={x_s: np.reshape(batch['x_s'], (BATCH, INPUT_DIM)), 
            y_: np.reshape(batch['y'], (BATCH, OUTPUT_DIM)), org_loss: np.reshape(batch['org_l'], (BATCH, 1)), 
            x_e: np.reshape(batch['x_e'], (BATCH, INPUT_DIM)), keep_prob: 1.0})
        print("step %d, training error %g"%(i, train_accuracy))
        if rerun == 0 and i == ITERS:
            print("done")
            #save_path = saver.save(sess, './interpolation-model')
            #print(save_path)
    gen = prediction.eval(feed_dict={x_s: np.reshape(batch['x_s'], (BATCH, INPUT_DIM)), 
        x_e: np.reshape(batch['x_e'], (BATCH, INPUT_DIM)), keep_prob: 1.0}, session=sess)
    train_step.run(feed_dict={x_s: np.reshape(batch['x_s'], (BATCH, INPUT_DIM)), 
        y_: np.reshape(batch['y'], (BATCH, OUTPUT_DIM)), org_loss: np.reshape(batch['org_l'], (BATCH, 1)), 
        x_e: np.reshape(batch['x_e'], (BATCH, INPUT_DIM)), keep_prob: 0.9, x: gen, y_class: })
    train_step_d.run(feed_dict={x: gen, y_class: })
    train_step_d.run(feed_dict={x: np.reshape(batch['x_s'], (BATCH, INPUT_DIM)). y_class: })
    


for j in range(len(shist[0])):
    resultimg = np.reshape(np.array(prediction.eval(feed_dict={x_s: np.reshape(shist[0][j], (1, INPUT_DIM)), 
        x_e: np.reshape(shist[1][j], (1, INPUT_DIM)), keep_prob: 1.0}, session=sess)), (HEIGHT, WIDTH))
    resultimg = np.clip(resultimg, 0, 255).astype(int)
    with open(str(j+1) + '.png', 'wb+') as f:
        w.write(f, np.reshape(resultimg, (-1, WIDTH)))
    print(j)