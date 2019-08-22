import itertools
import math
import random
import sys
from typing import List

import numpy as np
import png
import tensorflow as tf
from matplotlib import pyplot

import pflow_hs

class ImgData:
    """
    # h x w
    starts: List[np.ndarray]
    ends: List[np.ndarray]
    betweens: List[np.ndarray]
    diffs: List[float]
    # h x w x 2
    flows: List[np.ndarray]
    """
    pass

class ImgBatch:
    """
    # h x w
    x_s: List[np.ndarray]
    x_e: List[np.ndarray]
    y: List[np.ndarray]
    org_l: List[float]
    # h x w x 2
    flows: List[np.ndarray]
    """
    pass


WIDTH = 320
HEIGHT = 180
INPUT_DIM = WIDTH*HEIGHT
OUTPUT_DIM = INPUT_DIM
BATCH = 4
ITERS = 2048
NSIZE = 3 # Window size of optical flow
GEN_LEARN = 0.001
DISC_LEARN = 0.001

w = png.Writer(WIDTH, HEIGHT, greyscale=True, bitdepth=8)
np.set_printoptions(threshold=sys.maxsize)

def read_imagelist(filename: str) -> List[str]:
    imagelist = []
    with open(filename, 'r') as f:
        imagelist = f.readlines()
    imagelist = [im.strip() for im in imagelist]
    return imagelist

def gray(img: np.ndarray) -> np.ndarray:
    # h x w x 3 -> h x w
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def read_image(filename: str) -> np.ndarray:
    r = png.Reader(filename=filename)
    _, _, row, meta = r.read()
    row = np.array(list(map(np.uint16, row)))
    if np.product(np.shape(row)) == HEIGHT * WIDTH:
        return np.repeat(row[:, :, np.newaxis], 3, axis=2)
    row = np.reshape(list(map(np.uint16, list(row))), (HEIGHT, WIDTH, 3))
    return row

def read_images(imagelist: List[str]) -> ImgData:
    img_data = {"starts": [], "ends": [], "betweens": [], "diffs": [], "flows": []}

    for i in range(0, len(imagelist), 3):
        imstart = read_image(imagelist[i])
        imend = read_image(imagelist[i+2])
        imbetween = read_image(imagelist[i+1])
        
        imstart = gray(imstart)
        imend = gray(imend)
        imbetween = gray(imbetween)
        flow = pflow_hs.optical_flow(imstart, imend, NSIZE)

        img_data["starts"].append(imstart)
        img_data["ends"].append(imend)
        img_data["betweens"].append(imbetween)
        img_data["diffs"].append(np.sum(np.square(np.subtract(imend, imstart))))
        img_data["flows"].append(flow)
    return img_data

def weight_variable(shape: List[int]) -> tf.Variable:
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape: List[int]) -> tf.Variable:
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x: tf.Tensor, W: tf.Variable) -> tf.Tensor:
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d(x: tf.Tensor, W: tf.Variable, shape: List[int]) -> tf.Tensor:
    return tf.nn.conv2d_transpose(x, W, output_shape=shape, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x: tf.Tensor) -> tf.Tensor:
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def next_batch(img_data: ImgData, size: int) -> ImgBatch:
    indices = random.sample(range(len(img_data["starts"])), size)
    x_s = []
    x_e = []
    y = []
    org_l = []
    flows = []
    for index in indices:
        x_s.append(img_data["starts"][index])
        x_e.append(img_data["ends"][index])
        y.append(img_data["betweens"][index])
        org_l.append(img_data["diffs"][index])
        flows.append(img_data["flows"][index])
    return {
        'x_s': np.array(x_s, dtype=object),
        'x_e': np.array(x_e, dtype=object),
        'y': np.array(y, dtype=object),
        'org_l': np.array(org_l),
        'flows': np.array(flows)
    }

def make_labels(size: int, label: int) -> np.ndarray:
    to_return = np.zeros((size, 2))
    to_return[:, label] = 1
    return to_return

def train(img_data: ImgData, is_rerun: bool) -> None:
    x_s = tf.placeholder(tf.float32, shape=[None, INPUT_DIM], name='x_s')
    x_e = tf.placeholder(tf.float32, shape=[None, INPUT_DIM], name='x_e')
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM], name='y_')
    flow = tf.placeholder(tf.int32, shape=[None, HEIGHT, WIDTH, 2], name='flow')
    org_loss = tf.placeholder(tf.float32, shape=[BATCH, 1], name='org_loss')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    W_conv0 = weight_variable([5, 5, 1, 8])
    b_conv0 = bias_variable([8])
    x_l = tf.reshape(x_s, [-1, HEIGHT, WIDTH, 1])
    x_r = tf.reshape(x_e, [-1, HEIGHT, WIDTH, 1])
    h_conv0_l = tf.nn.relu(conv2d(x_l, W_conv0) + b_conv0)
    h_conv0_l_flow = tf.expand_dims(tf.gather_nd(h_conv0_l[0], flow[0]), 0)
    for i in range(1, BATCH):
        h_conv0_l_flow = tf.concat([h_conv0_l_flow, tf.expand_dims(tf.gather_nd(h_conv0_l[i], flow[i]), 0)], 0)
    h_pool0_l = max_pool_2x2(h_conv0_l_flow)
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
    y_conv = tf.reshape(tf.nn.relu(conv2d(h_depool2, W_reconv0) + b_reconv0), [-1, OUTPUT_DIM], name='y_conv')

    loss = tf.reduce_mean(tf.divide(tf.reduce_sum(tf.square(tf.subtract(y_, y_conv)), axis=1), org_loss))


    #Discriminator network
    x = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
    #binary classification
    y_class = tf.placeholder(tf.float32, shape=[None, 2])
    #If 1, then training generator, if 0, training discriminator
    train_type = tf.placeholder(tf.float32, shape=[None, 1])
    #Use y_conv if training generator, x if training discriminator
    true_input = x*(1 - train_type) + y_conv*train_type

    W_d_conv0 = weight_variable([5, 5, 1, 8])
    b_d_conv0 = bias_variable([8])

    h_d_conv0 = tf.nn.relu(conv2d(tf.reshape(true_input, [-1, HEIGHT, WIDTH, 1]), W_d_conv0) + b_d_conv0)
    h_d_pool0 = max_pool_2x2(h_d_conv0)

    W_d_conv1 = weight_variable([3, 3, 8, 16])
    b_d_conv1 = bias_variable([16])

    h_d_conv1 = tf.nn.relu(conv2d(h_d_pool0, W_d_conv1) + b_d_conv1)
    h_d_pool1 = max_pool_2x2(h_d_conv1)

    W_d_conv2 = weight_variable([3, 3, 16, 32])
    b_d_conv2 = bias_variable([32])

    h_d_conv2 = tf.nn.relu(conv2d(h_d_pool1, W_d_conv2) + b_d_conv2)

    h_d_r = tf.reshape(h_d_conv2, [-1, int(WIDTH/4 * HEIGHT/4 * 32)])
    W_d_fc0 = weight_variable([int(WIDTH/4 * HEIGHT/4 * 32), 512])
    b_d_fc0 = bias_variable([512])

    h_d_fc0 = tf.nn.relu(tf.matmul(h_d_r, W_d_fc0) + b_d_fc0)

    W_d_fc1 = weight_variable([512, 2])
    b_d_fc1 = bias_variable([2])

    out = tf.matmul(h_d_fc0, W_d_fc1) + b_d_fc1

    #discriminator loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_class, logits=out))
    train_step_d = tf.train.AdamOptimizer(DISC_LEARN).minimize(cross_entropy, var_list=[W_d_conv0, b_d_conv0, 
        W_d_conv1, b_d_conv1, W_d_conv2, b_d_conv2, W_d_fc0, b_d_fc0, W_d_fc1, b_d_fc1])

    #generator loss with discriminator loss included
    loss_with_d = loss - tf.log(cross_entropy + 1)
    train_step = tf.train.AdamOptimizer(GEN_LEARN).minimize(loss_with_d, var_list=[W_conv0, b_conv0, 
        W_conv1, b_conv1, W_conv2, b_conv2, W_deconv0, b_deconv0, W_deconv1, b_deconv1, 
        W_deconv2, b_deconv2, W_reconv0, b_reconv0])


    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    if is_rerun:
        saver.restore(sess, './models/interpolation-model')
        print("Model restored")
    else:
        sess.run(tf.global_variables_initializer())

    prediction = y_conv

    for i in range(ITERS+1):
        batch = next_batch(img_data, BATCH)
        if i % BATCH == 0:
            train_error, disc_loss = sess.run([loss, cross_entropy], feed_dict={
                x_s: np.reshape(batch['x_s'], (BATCH, INPUT_DIM)), 
                y_: np.reshape(batch['y'], (BATCH, OUTPUT_DIM)),
                org_loss: np.reshape(batch['org_l'], (BATCH, 1)), 
                x_e: np.reshape(batch['x_e'],(BATCH, INPUT_DIM)),
                flow: batch['flows'],
                keep_prob: 1.0,
                y_class: make_labels(BATCH, 1),
                x: np.zeros((BATCH, INPUT_DIM)), 
                train_type: np.ones((BATCH, 1))}
            )
            print("step %d, training error %g, disc loss %g" % (i, train_error, disc_loss))
            if not is_rerun and i == ITERS:
                print("done")
                #save_path = saver.save(sess, './models/interpolation-model')
                #print(save_path)
        gen = prediction.eval(feed_dict={
            x_s: np.reshape(batch['x_s'], (BATCH, INPUT_DIM)),
            x_e: np.reshape(batch['x_e'], (BATCH, INPUT_DIM)),
            flow: batch['flows'],
            keep_prob: 1.0
        })
        train_step.run(feed_dict={
            x_s: np.reshape(batch['x_s'], (BATCH, INPUT_DIM)),
            y_: np.reshape(batch['y'], (BATCH, OUTPUT_DIM)),
            org_loss: np.reshape(batch['org_l'], (BATCH, 1)),
            x_e: np.reshape(batch['x_e'], (BATCH, INPUT_DIM)),
            flow: batch['flows'],
            keep_prob: 0.9, 
            y_class: make_labels(BATCH, 1),
            x: gen, 
            train_type: np.ones((BATCH, 1))
        })
        train_step_d.run(feed_dict={
            x_s: np.reshape(batch['x_s'], (BATCH, INPUT_DIM)), 
            x_e: np.reshape(batch['x_e'], (BATCH, INPUT_DIM)),
            x: gen,
            flow: batch['flows'],
            y_class: make_labels(BATCH, 1),
            train_type: np.zeros((BATCH, 1))
        })
        train_step_d.run(feed_dict={
            x_s: np.reshape(batch['x_s'], (BATCH, INPUT_DIM)), 
            x_e: np.reshape(batch['x_e'], (BATCH, INPUT_DIM)),
            x: np.reshape(batch['y'], (BATCH, INPUT_DIM)),
            flow: batch['flows'],
            y_class: make_labels(BATCH, 0),
            train_type: np.zeros((BATCH, 1))
        })
        

    for j in range(len(img_data["starts"])):
        resultimg = np.reshape(
            np.array(
                prediction.eval(
                    feed_dict={
                        x_s: np.repeat(np.reshape(img_data["starts"][j], (1, INPUT_DIM)), BATCH, axis=0), 
                        x_e: np.repeat(np.reshape(img_data["ends"][j], (1, INPUT_DIM)), BATCH, axis=0),
                        flow: np.repeat(img_data["flows"][j][np.newaxis, ...], BATCH, axis=0),
                        keep_prob: 1.0
                    },
                    session=sess
                )
            )[0],
            (HEIGHT, WIDTH)
        )
        resultimg = np.clip(resultimg, 0, 255).astype(int)
        with open(str(j+1) + '.png', 'wb+') as f:
            w.write(f, np.reshape(resultimg, (HEIGHT, WIDTH)).tolist())
        print(j)

if __name__ == "__main__":
    imagelist = read_imagelist(sys.argv[1])
    img_data = read_images(imagelist)
    #pyplot.imshow(img_data["starts"][0], interpolation="nearest")
    #pyplot.show()
    #f = open('test.png', 'wb')
    #w.write(f, np.clip(img_data["starts"][0], 0, 255).astype(int).tolist())
    #f.close()
    train(img_data, False)
