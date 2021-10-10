import tensorflow as tf


def input_fn(filename, num_edge_types, batch_size=128, neg_size=1, nbr_size=1, num_epochs=1, min_after_times=32,
             capacity_times=64, shuffle=True, col_delim1=";", col_delim2=",", col_delim3=":"):
    def_vals_s = [[-1], [-1]] + [['']] + [['']] * 2 * num_edge_types + [[-1]] * num_edge_types
    def_vals_t = [[-1], [-1]] + [['']] + [['']] * 2 * num_edge_types + [[-1]] * num_edge_types
    def_vals = [[-1]] + def_vals_s + def_vals_t

    min_after_dequeue = min_after_times * batch_size
    capacity = capacity_times * batch_size
    data_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs, shuffle=shuffle)
    reader = tf.TextLineReader()
    _, value = reader.read_up_to(data_queue, batch_size)
    value = tf.train.shuffle_batch(
        [value],
        batch_size=batch_size,
        num_threads=24,
        capacity=capacity,
        enqueue_many=True,
        min_after_dequeue=min_after_dequeue)

    infos = tf.decode_csv(value, record_defaults=def_vals, field_delim=col_delim1)
    e_types = infos[0]
    s_ids = infos[1]
    s_types = infos[2]
    s_negs = decode_neg_ids(infos[3], neg_size, batch_size, col_delim2)

    s_nbr_infos = decode_nbr_infos(batch_size, nbr_size, infos[4:4 + num_edge_types],
                                   infos[4 + num_edge_types:4 + 2 * num_edge_types],
                                   infos[4 + 2 * num_edge_types:4 + 3 * num_edge_types])

    t_ids = infos[4 + 3 * num_edge_types]
    t_types = infos[5 + 3 * num_edge_types]
    t_negs = decode_neg_ids(infos[6 + 3 * num_edge_types], neg_size, batch_size, col_delim2)
    base = 7 + 3 * num_edge_types
    t_nbr_infos = decode_nbr_infos(batch_size, nbr_size, infos[base:base + num_edge_types],
                                   infos[base + num_edge_types:base + 2 * num_edge_types],
                                   infos[base + 2 * num_edge_types:base + 3 * num_edge_types])

    train_data = [e_types, [s_ids, s_types, s_negs, s_nbr_infos], [t_ids, t_types, t_negs, t_nbr_infos]]

    return train_data


def decode_nbr_infos(batch_size, nbr_size, ids_list, weights_list, flags):
    type_nbr_info = []
    for e_type in range(len(ids_list)):
        ids_str = ids_list[e_type]
        weight_str = weights_list[e_type]
        ids = decode_nbr_ids(ids_str, [batch_size, nbr_size])
        mask = decode_nbr_mask(ids_str, [batch_size, nbr_size])
        weights = decode_nbr_weights(weight_str, [batch_size, nbr_size])
        type_nbr_info.append([ids, mask, weights, flags[e_type]])
    return type_nbr_info


def decode_nbr_ids(str_tensor, shape, delim=","):
    sparse_info = tf.string_split(str_tensor, delimiter=delim)
    dense_matrix = tf.sparse_to_dense(sparse_info.indices, shape,
                                      tf.string_to_number(sparse_info.values, out_type=tf.int32), 0)
    return dense_matrix


def decode_nbr_mask(str_tensor, shape, delim=","):
    sparse_info = tf.string_split(str_tensor, delimiter=delim)
    dense_matrix = tf.sparse_to_dense(sparse_info.indices, shape, tf.ones_like(sparse_info.values, dtype=tf.int32),
                                      0)
    return dense_matrix


def decode_nbr_weights(str_tensor, shape, delim=","):
    sparse_info = tf.string_split(str_tensor, delimiter=delim)
    dense_matrix = tf.sparse_to_dense(sparse_info.indices, shape,
                                      tf.string_to_number(sparse_info.values, out_type=tf.float32), 0.0)
    return dense_matrix


def decode_neg_ids(str_tensor, neg_size, batch_size, delim=","):
    sparse_info = tf.string_split(str_tensor, delimiter=delim)
    return tf.sparse_to_dense(sparse_indices=sparse_info.indices,
                              sparse_values=tf.string_to_number(sparse_info.values, tf.int32),
                              output_shape=[batch_size, neg_size])
