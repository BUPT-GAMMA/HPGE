import os
import pickle as pkl

import tensorflow as tf

from data_loader import DataLoader
from evaluation import Evaluation
from input import input_fn
from model import HHP

flags = tf.flags
FLAGS = flags.FLAGS

# dataset
flags.DEFINE_string('path', '', 'path')
flags.DEFINE_string('outpath', '', 'outpath')
flags.DEFINE_string('graph', '', 'graph')
flags.DEFINE_string('labelfile', '', 'labelfile')
flags.DEFINE_float('ratio', 0.2, 'ratio')
flags.DEFINE_float('time_train', -1, 'time to split train and test')
flags.DEFINE_integer('init_data', -1, 'whether initialize the graph')
flags.DEFINE_string('task', 'NC', 'task: node classification, clustering and link prediction')
# model parameter
flags.DEFINE_integer('batch_size', 4096, 'batch_size')
flags.DEFINE_integer('neg_size', 5, 'neg_size')
flags.DEFINE_integer('nbr_size', 5, 'nbr_size')
flags.DEFINE_string('sample_type', 'important', 'cut_off or important')
flags.DEFINE_integer('num_epoch', 10, 'num_epoch')

flags.DEFINE_integer('node_dim', 128, 'node_dim')
flags.DEFINE_integer('num_node_type', 2, 'num_node_type')
flags.DEFINE_integer('num_edge_type', 4, 'num_edge_type')

flags.DEFINE_float('learning_rate', 0.0001, 'learning_rate')
flags.DEFINE_float('norm_rate', 0.001, 'norm_rate')

# gpu setting
flags.DEFINE_string('use_gpu', '0,1', 'use gpu')
flags.DEFINE_string('train', "train", 'train/test')

if FLAGS.use_gpu == '-1':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.use_gpu


def train(train_file, model_path, embedding_file, node_size):
    if os.path.exists(model_path):
        os.system('rm -rf {}'.format(model_path))
        os.mkdir(model_path)
    else:
        os.makedirs(model_path)
    data = input_fn(train_file, FLAGS.num_edge_type, FLAGS.batch_size, FLAGS.neg_size, FLAGS.nbr_size, FLAGS.num_epoch)

    global_step = tf.train.get_or_create_global_step()
    train_model = HHP(global_step, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.neg_size, FLAGS.nbr_size, node_size,
                      FLAGS.node_dim, FLAGS.num_node_type, FLAGS.num_edge_type, FLAGS.norm_rate, 'Adam')
    opts, loss = train_model.train(data)
    saver = train_model.init_saver()

    hooks = [tf.train.StopAtStepHook(last_step=8000000000)]
    scaffold = tf.train.Scaffold(saver=saver, init_op=tf.global_variables_initializer())

    with tf.train.MonitoredTrainingSession(scaffold=scaffold, hooks=hooks, checkpoint_dir=model_path,
                                           config=tf.ConfigProto(allow_soft_placement=True,
                                                                 log_device_placement=False)) as mon_sess:
        try:
            step = 0
            while not mon_sess.should_stop():
                step += 1
                _, loss_v = mon_sess.run([opts, loss])
                if step % 100 == 0:
                    print(step, loss_v)
                [embedding, edge_type_embed] = mon_sess.run([train_model.embedding, train_model.edge_type_embed])

        finally:
            pkl.dump([embedding, edge_type_embed], open(embedding_file, "wb"))


def test(embedding_file, task, ratio):
    eva = Evaluation(FLAGS.label_file, embedding_file, FLAGS.task)

    if task == "NC":
        eva.lr_classification(ratio)
    elif task == 'LP':
        eva.link_preds(ratio)
    elif task == 'CL':
        eva.kmeans()


def main(_):

    filename = FLAGS.graph
    data_loader = DataLoader(FLAGS.path, filename, delim=",", sample_type=FLAGS.sample_type, num_edge_types=FLAGS.num_edge_type, neg_size=FLAGS.neg_size, nbr_size=FLAGS.nbr_size)
    train_file = FLAGS.path + 'train_{}_{}_{}.csv'.format(FLAGS.nbr_size, FLAGS.neg_size, FLAGS.task)
    if FLAGS.init_data == 1:
        data_loader.generate_training_dataset(train_file)
    model_path = FLAGS.outpath + "HHP_{}_{}/".format(FLAGS.batch_size, FLAGS.num_epoch)
    embedding_file = model_path + "result.pkl"

    if FLAGS.train == 'train':
        print(data_loader.node_size)
        train(train_file, model_path, embedding_file, data_loader.node_size)
    else:
        test(embedding_file, FLAGS.task, FLAGS.ratio)


if __name__ == "__main__":
    tf.app.run(main)
