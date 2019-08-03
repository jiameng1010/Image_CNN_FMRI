import tensorflow as tf
tfgan = tf.contrib.gan
layers = tf.contrib.layers
import cv2
import numpy as np
import argparse
import importlib
import h5py
import os
import sys
from preprocessing.Data_fetcher import Data_fetcher

NUM_CUT = 6
DATA_DIR = '/home/mjia/Downloads/Image_CNN_FMRI/sceneViewingYork'
TRAINING_EPOCHES = 200
train_subject_list = np.arange(0, 20).tolist()
test_subject_list = np.arange(20, 35).tolist()

def main():
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['train'], './tf_memnet')
        writer = tf.summary.FileWriter("./Graph", sess.graph)
        writer.close()

        ops = sess.graph.get_operations()
        output_op = ops[170]
        mem_score = output_op.values()[0]

        last_feature_op = ops[160]
        last_feature = last_feature_op.values()[0]
        input = ops[0].values()

        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        pred = tf.layers.dense(last_feature, 2*NUM_CUT, kernel_regularizer=l2_regularizer, bias_regularizer=l2_regularizer, name="dense_add_by_mjia")
        gt = tf.placeholder(tf.float32, shape=pred.shape)
        loss = tf.reduce_mean(tf.losses.mean_squared_error(gt, pred))

        shuffled_kernel = tf.random.shuffle(sess.graph.get_tensor_by_name("dense_add_by_mjia/kernel:0"))
        shuffled_bias = tf.random.shuffle(sess.graph.get_tensor_by_name("dense_add_by_mjia/bias:0"))
        kernel_place_holder = tf.placeholder(tf.float32, shape=sess.graph.get_tensor_by_name("dense_add_by_mjia/kernel:0").shape)
        bias_place_holder = tf.placeholder(tf.float32, shape=sess.graph.get_tensor_by_name("dense_add_by_mjia/bias:0").shape)
        random_pred = tf.matmul(last_feature, kernel_place_holder)
        random_pred = tf.nn.bias_add(random_pred, bias_place_holder)
        rand_loss = tf.reduce_mean(tf.losses.mean_squared_error(gt, random_pred))

        train_variables = tf.trainable_variables()
        train_variables2 = train_variables[-2:]
        learning_rate = 0.00001
        batch = tf.Variable(0, trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate)
        train_op = trainer.minimize(loss, var_list=train_variables, global_step=batch)

        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        init = tf.initializers.variables(uninitialized_vars)
        sess.run(init)
        train_writer = tf.summary.FileWriter(os.path.join('./log', 'train'),sess.graph)


        '''##################################################################################################################'''
        '''##################################################################################################################'''
        '''##################################################################################################################'''
        '''##################################################################################################################'''
        '''##################################################################################################################'''
        def train_epoch(epoch_num):
            df = Data_fetcher(DATA_DIR, load_saved=False)
            num_batch = 0
            total_loss = 0.0
            #for subject in train_subject_list:
            for data in df.provide_epoch_one_subject(0):
                num_batch += 1
                feed_dict = {
                    input: data[0],
                    gt: data[1]
                }
                _, loss_val, predicted_activation = sess.run([train_op, loss, pred], feed_dict=feed_dict)

                total_loss += loss_val

            mean_loss = total_loss / num_batch

            print('***********************************************************************')
            print('epoch ' + str(epoch_num))
            print('Training Mean_Loss: ' + str(mean_loss))

        def eval_epoch(epoch_num):
            df = Data_fetcher(DATA_DIR, load_saved=False)
            num_batch = 0
            total_loss = 0.0
            #for subject in test_subject_list:
            for data in df.provide_epoch_one_subject(0, istrain=False):
                num_batch += 1
                feed_dict = {
                    input: data[0],
                    gt: data[1]
                }
                loss_val, predicted_activation = sess.run([loss, pred], feed_dict=feed_dict)
                total_loss += loss_val

            mean_loss = total_loss / num_batch
            print('Testing Mean_Loss: ' + str(mean_loss))

        def eval_epoch_rand():
            df = Data_fetcher(DATA_DIR, load_saved=False)
            num_batch = 0
            total_loss = 0.0
            random_kernel = sess.run(shuffled_kernel)
            random_bias = sess.run(shuffled_bias)
            #for subject in test_subject_list:
            for data in df.provide_epoch_one_subject(0, istrain=False):
                num_batch += 1
                feed_dict = {
                    kernel_place_holder: random_kernel,
                    bias_place_holder: random_bias,
                    input: data[0],
                    gt: data[1]
                }
                loss_val, predicted_activation = sess.run([rand_loss, pred], feed_dict=feed_dict)
                total_loss += loss_val

            mean_loss = total_loss / num_batch
            print('Random Testing Mean_Loss: ' + str(mean_loss))

        def correlation():
            df = Data_fetcher(DATA_DIR, load_saved=False)
            num_batch = 0
            total_loss = 0.0
            #for subject in test_subject_list:
            random_kernel = sess.run(shuffled_kernel)
            random_bias = sess.run(shuffled_bias)
            for data in df.provide_epoch_one_subject(0, istrain=False):
                num_batch += 1
                feed_dict = {
                    kernel_place_holder: random_kernel,
                    bias_place_holder: random_bias,
                    input: data[0],
                    gt: data[1]
                }
                mem_score_batch, predicted_activation_batch = sess.run([mem_score, random_pred], feed_dict=feed_dict)
                if num_batch == 1:
                    all_mem_score = mem_score_batch
                    all_pred_activation = predicted_activation_batch
                    all_gt = data[1]
                else:
                    all_mem_score = np.concatenate([all_mem_score, mem_score_batch], axis=0)
                    all_pred_activation = np.concatenate([all_pred_activation, predicted_activation_batch], axis=0)
                    all_gt = np.concatenate([all_gt, data[1]], axis=0)
            all_mem_score = all_mem_score - np.mean(all_mem_score, axis=0, keepdims=True)
            all_pred_activation = all_pred_activation - np.mean(all_pred_activation, axis=0, keepdims=True)
            all_gt = all_gt - np.mean(all_gt, axis=0, keepdims=True)
            correlation_memscore_activation = np.mean(all_mem_score*all_pred_activation)
            print(np.mean(all_mem_score*all_mem_score))
            print(np.mean(all_pred_activation*all_pred_activation))
            print(correlation_memscore_activation)
            print(correlation_memscore_activation/ np.sqrt(np.mean(all_pred_activation*all_pred_activation) * np.mean(all_mem_score*all_mem_score)))
            print('---------------------------------')

            correlation_memscore_gt = np.mean(all_mem_score * all_gt)
            print(np.mean(all_mem_score*all_mem_score))
            print(np.mean(all_gt*all_gt))
            print(correlation_memscore_gt)
            print(correlation_memscore_gt/ np.sqrt(np.mean(all_gt*all_gt)*np.mean(all_mem_score*all_mem_score)))
            print('---------------------------------')


        '''##################################################################################################################'''
        '''##################################################################################################################'''
        '''##################################################################################################################'''
        '''##################################################################################################################'''
        '''##################################################################################################################'''
        correlation()
        eval_epoch(0)
        eval_epoch_rand()
        for epoch in range(TRAINING_EPOCHES):
            train_epoch(epoch)
            eval_epoch(epoch)
            eval_epoch_rand()
            correlation()

        print("done")

if __name__ == '__main__':
    main()