# /*
# *Copyright (c) 2021, Alibaba Group;
# *Licensed under the Apache License, Version 2.0 (the "License");
# *you may not use this file except in compliance with the License.
# *You may obtain a copy of the License at

# *   http://www.apache.org/licenses/LICENSE-2.0

# *Unless required by applicable law or agreed to in writing, software
# *distributed under the License is distributed on an "AS IS" BASIS,
# *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *See the License for the specific language governing permissions and
# *limitations under the License.
# */
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.nn.rnn_cell import GRUCell
from tensorflow.keras.layers import Dense
from tensorflow.contrib import layers
from tensorflow.keras import Sequential
from module import MultiHeadAttention
from util import get_shape

class Model(object):
    def __init__(self, args, flag="DNN"):
        self.model_flag = flag
        self.reg = False
        self.batch_size = args.batch_size
        self.n_mid = args.n_mid
        self.boundary_cnt = 10
        self.args = args
        self.head = args.head
        self.n_cid = args.n_cid
        self.seq_len = args.maxlen

        self.metrics_op = {}
        self.m = args.m
        self.lamba = args.lamba
        self.embedding_dim = args.embedding_dim
        self.hidden_size = args.hidden_size

        with tf.name_scope('Inputs'):
            
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph') # user_id
            # positive category
            self.cid_batch_ph = tf.placeholder(tf.int32, [None, None], name='cid_batch_ph') #  [bs, 1] for train and [bs, N] for test
            self.s_batch_ph = tf.placeholder(tf.float32, [None,], name='s_batch_ph') # user cate score
            # negative category
            self.negc_batch_ph = tf.placeholder(tf.int32, [None, ], name='negc_batch_ph') # neg_cate_id [bs * neg_num,]
            # neighbor category
            self.neic_batch_ph = tf.placeholder(tf.int32, [None, None], name='neic_batch_ph') # neg_cate_id [bs, nei_num,]
            self.neis_batch_ph = tf.placeholder(tf.float32, [None, None], name='neis_batch_ph') # user neigh_cate score

            #user behaviour
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph') # history item id #[bs, seq_len]
            self.cid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cid_his_batch_ph') #[bs, seq_len]
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')

            # wide features
            self.widet_batch_ph = tf.placeholder(tf.int32, [None, None], name='wide_batch_h') #user interact with target cate_id [bs, N]
            self.wide_neg_batch_ph = tf.placeholder(tf.int32, [None, ], name='wide_neg_batch_h') #[bs*neg_num, ]
            self.wide_nei_batch_ph = tf.placeholder(tf.int32, [None, None], name='wide_nei_batch_h') #user interact with neighbor cate_id [bs, nei_num]

            self.lr = tf.placeholder(tf.float64, [])

        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [self.n_mid, self.embedding_dim], trainable=True)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

            self.cid_embedding_var = tf.get_variable("cid_embedding_var", [self.n_cid, self.embedding_dim], trainable=True)
            self.cid_batch_embedded = tf.nn.embedding_lookup(self.cid_embedding_var, self.cid_batch_ph) #[bs, dim]
            self.cid_his_batch_embedded = tf.nn.embedding_lookup(self.cid_embedding_var, self.cid_his_batch_ph) #[bs, seq_len, dim]
            self.negc_batch_embedded = tf.nn.embedding_lookup(self.cid_embedding_var, self.negc_batch_ph) #[bs*neg_num, dim]
            self.neic_batch_embedded = tf.nn.embedding_lookup(self.cid_embedding_var, self.neic_batch_ph) #[bs, nei_num, dim]

            self.wide_embedding_var = tf.get_variable("wide_embedding_var", [self.boundary_cnt, 8], trainable=True)
            self.wide_emb = {}
            self.wide_emb['pos'] = tf.nn.embedding_lookup(self.wide_embedding_var, self.widet_batch_ph) #[bs, dim]
            if get_shape(self.wide_emb['pos'])[-1] == 1:
                self.wide_emb['pos'] = tf.squeeze(self.wide_emb['pos'], -1)
            wide_neg_emb = tf.nn.embedding_lookup(self.wide_embedding_var, self.wide_neg_batch_ph) #[bs*neg_num, dim]
            self.wide_emb['neg'] = tf.tile(tf.expand_dims(wide_neg_emb, 0), [get_shape(self.wide_emb['pos'])[0], 1, 1])#[bs, bs*neg_num, dim]
            self.wide_emb['nei'] = tf.nn.embedding_lookup(self.wide_embedding_var, self.wide_nei_batch_ph) #[bs, nei_num, dim]

        self.cate_eb = self.cid_batch_embedded
        if get_shape(self.cate_eb)[-1] == 1:
           self.cate_eb = tf.squeeze(self.cate_eb, -1)
        self.cate_neg_eb = self.negc_batch_embedded
        self.cate_nei_eb = self.neic_batch_embedded

        self.item_his_eb = (self.mid_his_batch_embedded + self.cid_his_batch_embedded) * tf.reshape(self.mask, (-1, self.seq_len, 1))
    
    def user_net(self, ):
        """
        输入：
            1. 用户行为序列
        输出：
            user_embedding
        """
        # Multi-Head Attention
        # Input: self.item_his_eb: [batch_size, seq_len, dim]
        #       self.mask: [batch_size, seq_len]
        self_attention = MultiHeadAttention(h=self.head, d_v=self.embedding_dim, d_model=self.hidden_size)
        user_eb = self_attention.forward(queries=self.item_his_eb, keys=self.item_his_eb, values=self.item_his_eb, k_mask=self.mask, q_mask=None)
        user_eb = tf.reduce_sum(user_eb, axis=1)
        # MLP
        u_mlp = Sequential()
        u_mlp.add(Dense(self.hidden_size * 2, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
        u_mlp.add(Dense(self.hidden_size))
        user_eb = u_mlp(user_eb)
        return user_eb

    
    def cate_net(self, ):
        """
        Input: 
        Output: cate_embedding
        """
        c_mlp = Sequential()
        c_mlp.add(Dense(self.hidden_size * 2, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
        c_mlp.add(Dense(self.hidden_size))

        cate_eb_list = {}
        cate_eb_list['pos'] = c_mlp(self.cate_eb)
        cate_eb_list['neg'] = c_mlp(self.cate_neg_eb)
        cate_eb_list['nei'] = c_mlp(self.cate_nei_eb)
        return cate_eb_list

    def wide_net(self, user_eb, cate_eb_list):
        """
        Input: 
            1. InnerProduct(user_embedding, cate_embedding)
            2. wide embedding
        Outputs: 
            logits = f(concat(InnerProduct + wide embedding))
            including [pos, neg, nei]
        """
        wide_mlp = Sequential()
        wide_mlp.add(Dense(1))

        self.deep_logits = {}
        if len(get_shape(cate_eb_list['pos'])) == 2: #train model
            # [bs,dim] [bs,dim, 1]
            self.deep_logits['pos'] = tf.reduce_sum(tf.multiply(user_eb, cate_eb_list['pos']), axis=1, keep_dims=True) #[batch_size, 1]
        else:
            # [bs,dim] [bs, N, dim]
            self.deep_logits['pos'] = tf.matmul(cate_eb_list['pos'], tf.expand_dims(user_eb, 2)) #[bs, N, 1]
        #[bs, dim] [bs*neg_num, dim]
        self.deep_logits['neg'] = tf.expand_dims(tf.matmul(user_eb, cate_eb_list['neg'], transpose_b=True), 2) #[bs, bs*neg, 1]
        #[bs,dim, 1] [bs, nei_num, dim]
        self.deep_logits['nei'] = tf.matmul(cate_eb_list['nei'], tf.expand_dims(user_eb, 2)) #[bs, nei_num, 1]

        """
        wide_emb[pos]: [bs, dim] / [bs, N, dim]
                [neg]: [bs, bs*neg_num, dim]
                [nei]: [bs, nei_num, dim]
        """
        self.logits = {}
        self.logits['pos'] = wide_mlp(tf.concat(values=[self.deep_logits['pos'], self.wide_emb['pos']], axis=-1))
        self.logits['neg'] = wide_mlp(tf.concat(values=[self.deep_logits['neg'], self.wide_emb['neg']], axis=-1))
        self.logits['nei'] = wide_mlp(tf.concat(values=[self.deep_logits['nei'], self.wide_emb['nei']], axis=-1))

    def build_loss(self, logits):
        #[bs, seq_len, dim]
        #[bs, seq_len]
        #[bs, dim]
        #[bs*neg, dim]
        ce_loss = self.build_sampled_softmax_loss(logits)
        t_loss = self.build_triple_loss(logits)
        self.loss = ce_loss + self.lamba * t_loss

        self.metrics_op['ce_loss'] = ce_loss
        self.metrics_op['t_loss'] = t_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        

    def build_sampled_softmax_loss(self, logits):
        pos_logits = logits['pos']
        neg_logits = logits['neg']

        labels_pos = tf.ones_like(pos_logits, dtype=tf.int64)  # [bs, 1]
        labels_neg = tf.zeros_like(neg_logits, dtype=tf.int64)  # [bs, neg_num * bs]
        labels = tf.concat([labels_pos, labels_neg], 1)  # [bs, neg_num * bs + 1]

        logits = tf.cast(tf.concat([pos_logits, neg_logits], 1), dtype=tf.float32) #[bs, neg_num * bs + 1]
        labels = tf.cast(labels, dtype=tf.float32)

        ce_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) #[bs]
        loss = tf.reduce_mean(ce_loss)
        return loss
    
    def build_triple_loss(self, logits):
        # 1. Prepare Data
        logit = logits['pos'] #[bs, 1]
        neigh_logit = tf.squeeze(logits['nei'], -1) # [bs, nei, 1]
        label = tf.expand_dims(self.s_batch_ph ,1) #[bs, 1]
        neigh_label = self.neis_batch_ph #[bs, nei]

        # 2. Calculate Loss
        triplet_loss = tf.multiply(tf.sign(label - neigh_label), neigh_logit - logit) #[bs, 5]

        # 3. Select semi_hard samples
        triplet_loss = triplet_loss + self.m
        # 4. Filter easy sample
        triplet_loss = tf.maximum(triplet_loss, 0.0)
        # 5. Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)

        # 6. Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
        return triplet_loss

        
    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0], #user_id
            self.cid_batch_ph: inps[1], #target_cate_id
            self.s_batch_ph:inps[2], #user - cate score
            self.widet_batch_ph: inps[3],
            self.mid_his_batch_ph: inps[4], #(hist_item_ids)
            self.cid_his_batch_ph: inps[5], #(hist_cate_ids)
            self.mask: inps[6], #mask [1.0, xxx]
            self.negc_batch_ph: inps[7],
            self.wide_neg_batch_ph: inps[8],
            self.neic_batch_ph: inps[9],
            self.neis_batch_ph: inps[10],
            self.wide_nei_batch_ph: inps[11],
            self.lr: inps[12] #learning rate
        }

        loss, _, metrics = sess.run([self.loss, self.optimizer, self.metrics_op], feed_dict=feed_dict)
        return loss, metrics

    
    def output(self, sess, inps):
        logits = sess.run(self.logits['pos'], feed_dict={
            self.uid_batch_ph: inps[0], #user_id
            self.cid_batch_ph: inps[1], #target_cate_id with [batch_size, N]
            self.widet_batch_ph: inps[2],
            self.mid_his_batch_ph: inps[3], #(hist_item_ids)
            self.cid_his_batch_ph: inps[4], #(hist_cate_ids)
            self.mask: inps[5] #mask [1.0, xxx]
        })

        return logits
    
    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)

class Model_U2C(Model):
    def __init__(self, args):
        super(Model_U2C, self).__init__(args, flag="U2C")
        # TODO: Load deep parameter only
        user_emb = self.user_net()
        cate_emb_list = self.cate_net()
        self.wide_net(user_emb, cate_emb_list)
        self.build_loss(self.logits)
        

def get_model(dataset, args):
    if args.model_type == 'U2C':
        model = Model_U2C(args)
    else:
        print ("Invalid model_type : %s", model_type)
        return
    return model