from tools import *


class HHP:

    def __init__(self, global_step, learning_rate, batch_size, neg_size, nbr_size, node_size, node_dim, num_node_types,
                 num_edge_types,
                 norm_rate, opt='Adam'):
        # init parameters
        self.global_step = global_step
        self.node_size = node_size
        self.node_dim = node_dim
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.batch_size = batch_size
        self.neg_size = neg_size
        self.norm_rate = norm_rate
        self.learning_rate = learning_rate
        self.nbr_size = nbr_size
        self.l2_regular = tf.contrib.layers.l2_regularizer(self.norm_rate)

        with tf.variable_scope('parameters', reuse=tf.AUTO_REUSE):
            init_range_embed = np.sqrt(3.0 / (self.node_size + self.node_dim))
            self.embedding = tf.get_variable('embedding_table',
                                             initializer=tf.random_uniform([self.node_size, self.node_dim],
                                                                           minval=-init_range_embed,
                                                                           maxval=init_range_embed, dtype=tf.float32))

            init_range_edge_type = np.sqrt(3.0 / (self.node_dim + self.node_dim))
            self.edge_type_embed = tf.get_variable('edge_type_r',
                                                   initializer=tf.random_uniform(
                                                       [self.num_edge_types, self.node_dim],
                                                       minval=-init_range_edge_type,
                                                       maxval=init_range_edge_type,
                                                       dtype=tf.float32))
            if opt == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            else:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def train(self, batch_data):
        e_types = batch_data[0]

        s_ids, s_types, s_negs, s_nbr_infos = batch_data[1]
        t_ids, t_types, t_negs, t_nbr_infos = batch_data[2]

        basic_info = self.construct_mu(s_ids, s_types, t_ids, t_types, e_types, t_negs, s_negs, neg_size=self.neg_size)
        mu, neg_mus_st, neg_mus_ts, s_embed, t_embed, neg_embed_s_list, neg_embed_t_list = basic_info
        pos_loss_st, neg_loss_st = self.construct_mutual_influence(s_embed, s_nbr_infos, t_embed, neg_embed_t_list)
        pos_loss_ts, neg_loss_ts = self.construct_mutual_influence(t_embed, t_nbr_infos, s_embed, neg_embed_s_list)

        lambda_st_pos = mu + pos_loss_st + pos_loss_ts
        lambda_st_neg = neg_mus_st + neg_loss_st
        lambda_ts_neg = neg_mus_ts + neg_loss_ts
        #
        # lambda_st_pos = mu
        # lambda_st_neg = neg_mus_st
        # lambda_ts_neg = neg_mus_ts


        loss = -tf.reduce_mean(tf.log(tf.sigmoid(lambda_st_pos) + 1e-6)) \
               - tf.reduce_mean(tf.log(tf.sigmoid(-lambda_st_neg) + 1e-6)) \
               - tf.reduce_mean(tf.log(tf.sigmoid(-lambda_ts_neg) + 1e-6)) \
               + self.norm_rate * tf.reduce_sum(tf.pow(self.edge_type_embed, 2))
        opts = self.optimizer.minimize(loss, global_step=self.global_step)
        return [opts, loss]

    def construct_node_latent_embed(self, node_ids, node_types, node_size, type_size):
        # return tf.gather(self.embedding, node_ids)
        with tf.variable_scope('node_type_embed', reuse=tf.AUTO_REUSE):
            indices = tf.range(node_size) * type_size + node_types
            embedding = tf.gather(self.embedding, node_ids)
            new_matrix = tf.reshape(tf.unsorted_segment_sum(embedding, indices, node_size * type_size),
                                    [node_size, type_size, self.node_dim])
            embed_typed = tf.layers.dense(new_matrix, self.node_dim, activation=tf.nn.leaky_relu,
                                          kernel_regularizer=self.l2_regular, reuse=tf.AUTO_REUSE,
                                          name='node_type_embed')

            node_final_embeds = tf.gather(tf.reshape(embed_typed, [node_size * type_size, self.node_dim]), indices)
            return node_final_embeds

    def construct_mu(self, s_ids, s_types, t_ids, t_types, e_types, t_neg_ids, s_neg_ids, neg_size):
        with tf.variable_scope('mu_layer', reuse=tf.AUTO_REUSE):
            s_embed = self.construct_node_latent_embed(s_ids, s_types, self.batch_size, self.num_node_types)
            t_embed = self.construct_node_latent_embed(t_ids, t_types, self.batch_size, self.num_node_types)
            e_embed = tf.gather(self.edge_type_embed, e_types)
            mu = self.g_func(s_embed + e_embed, t_embed, 'l2')
            neg_mus_st = []
            neg_embed_s_list = []
            neg_embed_t_list = []
            neg_mus_ts = []
            for i in range(neg_size):
                neg_t_embed = self.construct_node_latent_embed(t_neg_ids[:, i], t_types, self.batch_size,
                                                               self.num_node_types)
                neg_embed_t_list.append(neg_t_embed)
                neg_mu_t_i = tf.reshape(self.g_func(s_embed, neg_t_embed, 'l2'), [-1, 1])
                neg_mus_st.append(neg_mu_t_i)
                neg_s_embed = self.construct_node_latent_embed(s_neg_ids[:, i], s_types, self.batch_size,
                                                               self.num_node_types)
                neg_embed_s_list.append(neg_s_embed)
                neg_mu_s_i = tf.reshape(self.g_func(t_embed, neg_s_embed, 'l2'), [-1, 1])
                neg_mus_ts.append(neg_mu_s_i)

            return mu, tf.concat(neg_mus_st, axis=-1), tf.concat(neg_mus_ts, axis=-1), \
                   s_embed, t_embed, neg_embed_s_list, neg_embed_t_list

    def construct_mutual_influence(self, node_embed, node_nbr_infos, target_embed, neg_embed):

        with tf.variable_scope('multual_influence', reuse=tf.AUTO_REUSE):
            pos_info = []
            neg_info = []
            att_info = []
            mask = []

            for i in range(self.num_edge_types):
                nbr_ids, nbr_masks, nbr_weights, nbr_flag = node_nbr_infos[i]

                pos_g, neg_g, hete_att = tf.cond(
                    tf.reduce_all(nbr_flag > 0),
                    false_fn=lambda: [tf.zeros(shape=[self.batch_size, 1], dtype=tf.float32),
                                      tf.zeros(shape=[self.batch_size, self.neg_size], dtype=tf.float32),
                                      tf.zeros(shape=[self.batch_size, 1], dtype=tf.float32)],
                    true_fn=lambda: self.edge_type_distance(node_embed, nbr_ids, i, nbr_weights,
                                                            nbr_masks, target_embed, neg_embed)
                )

                pos_info.append(pos_g)
                neg_info.append(neg_g)
                att_info.append(hete_att)
                mask.append(nbr_flag)

            mask = tf.cast(tf.reshape(tf.concat(mask, axis=-1), [self.batch_size, self.num_edge_types]), tf.bool)
            att_info = tf.concat(att_info, axis=-1)
            padding = tf.fill([self.batch_size, self.num_edge_types], -2 ** 32 + 1.0)
            padding2 = tf.fill([self.batch_size, self.num_edge_types], 0.0)
            att_v1 = tf.nn.softmax(tf.where(mask, att_info, padding), axis=1)
            norm_att = tf.nn.softmax(tf.where(mask, att_v1, padding2), axis=1)

            pos_info = tf.concat(pos_info, axis=-1)
            neg_info = tf.reshape(tf.concat(neg_info, axis=-1), [self.batch_size, self.num_edge_types, self.neg_size])
            neg_info = tf.transpose(neg_info, [0, 2, 1])

            pos_loss = tf.reduce_sum(tf.multiply(norm_att, pos_info), axis=-1)
            neg_loss = tf.reduce_sum(tf.matmul(neg_info, tf.expand_dims(norm_att, axis=2)), axis=-1)
            return pos_loss, neg_loss

    def edge_type_distance(self, node_embed, ids, e_type, weight, mask, target_embed, neg_embed):
        with tf.variable_scope('multual_influence_{}'.format(e_type), reuse=tf.AUTO_REUSE):
            nbr_embed = tf.layers.dense(tf.gather(self.embedding, ids), self.node_dim, reuse=tf.AUTO_REUSE,
                                        kernel_regularizer=self.l2_regular, name="edge_type_{}".format(e_type))
            edge_embed = tf.reshape(tf.gather(self.edge_type_embed, [e_type]), (1, 1, self.node_dim))
            node_embed = tf.expand_dims(node_embed, axis=1)
            nbr_distance = self.g_func(node_embed + edge_embed, nbr_embed, opt='l2')
            paddings = tf.fill(tf.shape(nbr_distance), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
            paddings2 = tf.fill(tf.shape(nbr_distance), tf.constant(0, dtype=tf.float32))
            nbr_distance2 = tf.where(tf.cast(mask, dtype=tf.bool), nbr_distance, paddings)
            atts = tf.nn.softmax(nbr_distance2, axis=-1)
            atts_2 = tf.where(tf.cast(mask, dtype=tf.bool), atts, paddings2)
            new_weight = tf.multiply(atts_2, weight)

            mutual_subs = self.g_func(nbr_embed, tf.expand_dims(target_embed, axis=1), 'l2')
            mutual_neg_subs = [self.g_func(nbr_embed, tf.expand_dims(neg_embed[i], axis=1), 'l2') for i in
                               range(self.neg_size)]

            avg_embed = tf.reduce_sum(tf.matmul(tf.expand_dims(atts_2, 1), nbr_embed), axis=1)
            avg_weight_1 = tf.reduce_sum(weight, axis=1)
            nbr_numbers = tf.clip_by_value(tf.cast(tf.reduce_sum(mask, axis=1), tf.float32), 1.0, self.nbr_size)
            ave_weight = tf.reshape(avg_weight_1 / nbr_numbers, [-1, 1])

            avg_info = tf.multiply(ave_weight, avg_embed)

            hete_att = tf.layers.dense(avg_info, 1, tf.nn.leaky_relu, kernel_regularizer=self.l2_regular,
                                       reuse=tf.AUTO_REUSE, name='hete_att_{}'.format(e_type))
            pos_mutual_influ = tf.reduce_sum(tf.multiply(new_weight, mutual_subs), axis=-1)
            neg_mutual_influ = [
                tf.reshape(tf.reduce_sum(tf.multiply(new_weight, mutual_neg_subs[i]), axis=-1), [self.batch_size, 1])
                for i in
                range(self.neg_size)]
            return [tf.reshape(pos_mutual_influ, [self.batch_size, 1]), \
                    tf.concat(tf.reshape(neg_mutual_influ, [self.batch_size, self.neg_size]), axis=1), \
                    tf.reshape(hete_att, [self.batch_size, 1])]

    def g_func(self, x, y, opt='l2'):
        if opt == 'l2':
            return -tf.reduce_sum((x - y) ** 2, axis=-1)
        elif opt == 'l1':
            return -tf.reduce_sum(tf.abs(x - y), axis=-1)
        else:
            return -tf.reduce_sum((x - y) ** 2, axis=-1)

    def init_saver(self):
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        bn_moving_vars += [g for g in g_list if 'global_step' in g.name]
        self.saver = tf.train.Saver(var_list=var_list + bn_moving_vars, max_to_keep=1)
