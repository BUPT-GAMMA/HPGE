import pickle as pkl

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

class Evaluation:

    def __init__(self, label_file=None, embedding_file=None, task=None):
        self.label_file = label_file
        self.embedding_file = embedding_file
        self.task = task

    def lr_classification(self, train_ratio):
        ids, labels, _ = pkl.load(open(self.label_file, "rb"))
        node_embed, _ = pkl.load(open(self.embedding_file, "rb"))

        features = node_embed[ids]
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=1 - train_ratio,random_state=9)

        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_valid_pred = lr.predict(x_valid)
        micro_f1 = f1_score(y_valid, y_valid_pred, average='micro')
        macro_f1 = f1_score(y_valid, y_valid_pred, average='macro')
        print('Macro_F1_score:{}'.format(macro_f1))
        print('Micro_F1_score:{}'.format(micro_f1))

    def kmeans(self):
        ids, labels, _ = pkl.load(open(self.label_file, "rb"))
        ids = np.array(ids)
        labels = np.array(labels)
        node_embed, _ = pkl.load(open(self.embedding_file, "rb"))
        features = node_embed[ids]
        km = KMeans(n_clusters=8, max_iter=1000)
        preds = km.fit_predict(features)
        nmi = normalized_mutual_info_score(labels, preds)
        ari = adjusted_rand_score(labels, preds)
        print('NMI:{}'.format(nmi))
        print('ARI:{}'.format(ari))

    def link_preds(self, train_ratio):
        labels, left_ids, right_ids, edge_types = pkl.load(open(self.label_file, "rb"))
        node_embed, edge_type_embedding = pkl.load(open(self.embedding_file, "rb"))

        left_embed = node_embed[left_ids]
        right_embed = node_embed[right_ids]
        edge_embed = edge_type_embedding[edge_types]
        features = [left_embed + edge_embed - right_embed]
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=1 - train_ratio,
                                                              random_state=9)
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_valid_pred_prob = lr.predict_proba(x_valid)[:, 1]
        y_valid_pred_01 = lr.predict(x_valid)
        acc = accuracy_score(y_valid, y_valid_pred_01)
        auc = roc_auc_score(y_valid, y_valid_pred_prob)
        f1 = f1_score(y_valid, y_valid_pred_01)
        print('acc:{}'.format(acc))
        print('auc:{}'.format(auc))
        print('f1:{}'.format(f1))
