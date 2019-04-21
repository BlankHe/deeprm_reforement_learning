import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(self, pa, output_graph=False):

        self.pa = pa
        self.train_mode = True
        self.n_actions = self.pa.network_output_dim
        self.input_height = self.pa.network_input_height
        self.input_width = self.pa.network_input_width
        self.flat_high = int(math.ceil(math.ceil(self.input_height / 2.0) / 2.0))
        self.flat_width = int(math.ceil(math.ceil(self.input_width / 2.0) / 2.0))
        self.lr = self.pa.lr_rate
        self.gamma = 0.95

        self.flat = [None] * self.pa.num_res
        self.dropout = [None] * self.pa.num_res
        self.all_act = [None] * self.pa.num_res
        self.all_act_prob = [None] * self.pa.num_res
        self.neg_log_prob = [None] * self.pa.num_res

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_cnn_net()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.input_height, self.input_width], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            # name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            # name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,
                                                                          labels=self.tf_acts)  # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            # self.log = -tf.log(self.all_act_prob)
            # self.one_hot = tf.one_hot(self.tf_acts, self.n_actions)
            # self.reduce_sum = self.log * self.one_hot
            # self.neg_log_prob = tf.reduce_sum(self.reduce_sum, axis=1)
            # self.mean = self.neg_log_prob * self.tf_vt
            # self.loss = tf.reduce_mean(self.mean)  # reward guided loss
            self.loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _build_cnn_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, 1], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, self.pa.num_res], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=self.tf_obs,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same')

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='same')

        # Convolutional Layer #3 and Pooling Layer #3

        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=self.pa.num_res,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.relu)

        conv3_flat = tf.reshape(conv3, [-1, self.flat_high * self.flat_width, self.pa.num_res])

        for m in range(self.pa.num_res):
            self.flat[m] = conv3_flat[:, :, m]
            self.dropout[m] = tf.layers.dropout(inputs=self.flat[m], rate=0.4, training=self.train_mode)
            self.all_act[m] = tf.layers.dense(inputs=self.dropout[m],
                                              units=self.n_actions,
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                              bias_initializer=tf.constant_initializer(0.1))
            self.all_act_prob[m] = tf.nn.softmax(self.all_act[m], name='act_prob')

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            for m in range(self.pa.num_res):
                self.neg_log_prob[m] = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act[m],
                                                                                      labels=self.tf_acts[:,
                                                                                             m])  # this is negative log of chosen action

                # or in this way:
                # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
                # self.log = -tf.log(self.all_act_prob)
                # self.one_hot = tf.one_hot(self.tf_acts, self.n_actions)
                # self.reduce_sum = self.log * self.one_hot
                # self.neg_log_prob = tf.reduce_sum(self.reduce_sum, axis=1)
                # self.mean = self.neg_log_prob * self.tf_vt
                # self.loss = tf.reduce_mean(self.mean)  # reward guided loss
            neg_log_prob = tf.concat([self.neg_log_prob[i] for i in range(self.pa.num_res)], 0)
            tf_vt_boardcast = tf.concat([self.tf_vt for i in range(self.pa.num_res)], 0)
            self.loss = tf.reduce_mean(neg_log_prob * tf_vt_boardcast)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        action = []
        self.train_mode = False
        for i in range(self.pa.num_res):
            prob_weights = self.sess.run(self.all_act_prob[i], feed_dict={self.tf_obs: observation})
            a = np.random.choice(range(prob_weights.shape[1]),
                                 p=prob_weights.ravel())  # select action w.r.t the actions prob
            action.append(a)

        return action

    def store_ob(self, s):
        self.ep_obs.append(s)

    def store_action(self, a):
        self.ep_as.append(a)

    def store_adv(self, r):
        self.ep_rs.append(r)

    def learn(self, all_ob, all_action, all_adv):


        self.train_mode = True
        # discount and normalize episode reward
        # discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        # _, loss, log, one_hot, reduce_sum, neg_log_prob, mean = self.sess.run([self.train_op, self.loss,
        # self.log,
        # self.one_hot,
        # self.reduce_sum,
        # self.neg_log_prob,
        # self.mean],
        # feed_dict={
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            # self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            # self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            # self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
            self.tf_obs: np.array(all_ob),  # shape=[None, n_obs]
            self.tf_acts: np.array(all_action),  # shape=[None, ]
            self.tf_vt: np.array(all_adv),  # shape=[None, ]
        })


        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        #print(loss)
        return loss

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        # discounted_ep_rs = np.zeros_like(self.ep_rs)
        discounted_ep_rs = np.fabs(np.array(self.ep_rs))

        # running_add = 0
        # for t in reversed(range(0, len(self.ep_rs))):
        # running_add = running_add * self.gamma + self.ep_rs[t]
        # discounted_ep_rs[t] = running_add

        # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_data(self, pg_resume):
        self.saver.save(self.sess, pg_resume + '.ckpt')

    def load_data(self, pg_resume):
        self.saver.restore(self.sess, pg_resume)
