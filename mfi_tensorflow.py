import numpy as np
import tensorflow as tf
import tflearn
import pickle
from get_data import cities
from regression_to_classification import simple_regression_to_classsification as rtc
from regression_to_classification import classification_to_regression as ctr
from get_distance_with_txt import distance, build_kernel

tf.app.flags.DEFINE_string("train_test", "train", "train_or_test")
FLAGS = tf.app.flags.FLAGS
batch_size = 256
city_nums = len(cities)
MAX_ITR = 5
print('building')
class_nums = 40
Qin = tf.placeholder(dtype=tf.float32, shape=(batch_size, city_nums, 40))
K1 = tf.placeholder(tf.float32, (batch_size, city_nums, city_nums))
K2 = tf.placeholder(tf.float32, (batch_size, city_nums, city_nums))
U = -tf.log(Qin)
Q = Qin
MAX_ITR = 5

for itr in range(MAX_ITR):
    # message passing
    print(itr + 1)
    Q1 = tf.matmul(K1, Q)
    Q2 = tf.matmul(K2, Q)
   
    # weight filter
    Q1_ = tflearn.layers.conv_1d(incoming=Q1, nb_filter=class_nums,
                                filter_size=1,
                                reuse=tf.AUTO_REUSE, name="weight_filter1",
                                scope="weight_filter1")
    Q2_ = tflearn.layers.conv_1d(incoming=Q2, nb_filter=class_nums,
                                filter_size=1,
                                reuse=tf.AUTO_REUSE, name="weight_filter2",
                                scope="weight_filter2")

    Q = tf.nn.softmax(-U-Q1_-Q2_)
print("built")
saver = tf.train.Saver()
counter = 0
if FLAGS.train_test == "train":
    Label = tf.placeholder(tf.int32, shape=(batch_size, city_nums, 40))
    Loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Q, labels=Label))
    train_step = tf.train.AdamOptimizer(0.001).minimize(Loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.graph.finalize()
        sess.run(init)
        epoch = 0
        saver.save(sess, "model/", global_step=epoch)
        while True:
            sloss = 0.0
            data = pickle.load(open("data/mfi_valid_batch", "rb"))
            np.random.shuffle(data)
            by = []
            bq = []
            bf = []
            for y, q, f, avg_q in data:
                y_as_label = [rtc(y[i], step=2) for i in range(y.shape[0])]
                by.append(np.array(y_as_label))
                bq.append(q)
                bf.append(f)
                if len(by) == batch_size:
                    k1 = np.array([build_kernel(bf[i]) for i in range(batch_size)])
                    k2 = np.array([build_kernel() for i in range(batch_size)])
                    _, loss = sess.run([train_step, Loss],
                                       feed_dict={Qin: np.array(bq),
                                                  Label: np.array(by),
                                                  K1: k1,
                                                  K2: k2})
                    sloss += loss
                    by = []
                    bf = []
                    bq = []
            epoch += 1
            print("ave loss is {} at epoch {}".format(sloss/len(data), epoch))
            saver.save(sess, "model/", global_step=epoch)
            if epoch % 10 == 0:
                test_data = pickle.load(open("data/mfi_test_batch", "rb"))
                before_se = 0.0
                after_se = 0
                total = 0
                by = []
                bq = []
                bf = []
                bavg = []
                bt = []
                for y, q, f, avg_q in test_data:
                    y_as_label = [rtc(y[i], step=2) for i in range(y.shape[0])]
                    by.append(np.array(y_as_label))
                    bq.append(q)
                    bf.append(f)
                    bavg.append(avg_q)
                    bt.append(y)
                    if len(by) == batch_size:
                        k1 = np.array([build_kernel(bf[i]) for i in range(batch_size)])
                        k2 = np.array([build_kernel() for i in range(batch_size)])
                        qout, = sess.run([Q,],
                                    feed_dict={Qin: np.array(bq),
                                               K1: k1,
                                               K2: k2})
                        for b in range(batch_size):
                            q_result = [ctr(qout[b][c]) for c in range(len(cities))] #[tmps for city in cities]
                            for c in range(len(cities)):
                                after_r = q_result[c]
                                before_r = bavg[b][c]
                                true_value = bt[b][c]
                                before_se +=  (before_r - true_value) ** 2
                                after_se += (after_r - true_value) ** 2
                                total += 1
                        by = []
                        bf = []
                        bq = []
                        bavg = []
                        bt = []
                print(before_se/total, after_se/total, epoch)
'''
else:
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state("model/")
        saver.restore(sess, ckpt.model_checkpoint_path)
        data = pickle.load(open("mfi_test_batch", "rb"))
        before_se = 0.0
        after_se = 0
        total = 0
        data = pickle.load(open("data/mfi_test_batch", "rb"))
        by = []
        bq = []
        bf = []
        bavg = []
        bt = []
        for y, q, f, avg_q in data:
            y_as_label = [rtc(y[i], step=2) for i in range(y.shape[0])]
            by.append(np.array(y_as_label))
            bq.append(q)
            bf.append(f)
            bavg.append(avg_q)
            bt.append(y)
            if len(by) == batch_size:
                qout = sess.run([Q,],
                               feed_dict={Qin: np.array(bq),
                                          feature_cache: np.array(bf)})
                for b in range(batch_size):
                    q_result = ctr(qout[b]) #[tmps for city in cities]
                    for c in range(len(cities)):
                        after_r = ctr(q_result[c])
                        before_r = bavg[b][c]
                        true_value = bt[b][c]
                        before_se +=  (before_r - true_value) ** 2
                        after_se += (after_r - true_value) ** 2
                        total += 1
                by = []
                bf = []
                bq = []
                bavg = []
                bt = []
        print(before_se/total, after_se/total)
'''