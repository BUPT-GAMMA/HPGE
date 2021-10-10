import numpy as np
import pandas as pd
import pickle
import networkx as nx
import random
#
confLabel = {'AAAI': 2, 'CIKM': 3, 'CVPR': 2, 'ECIR': 3, 'ECML': 2, 'EDBT': 0, 'ICDE': 0, 'ICDM': 1, 'ICML': 2,
             'IJCAI': 2, 'KDD': 1, 'PAKDD': 1, 'PKDD': 1, 'PODS': 0, 'SDM': 1, 'SIGIR': 3, 'SIGMOD Conference': 0,
             'VLDB': 0, 'WWW': 3, 'WSDM': 3}

# 读入文件
# 为author和conf重新编码
min_year = 1969
max_year = 2015
dealt_year = max_year - min_year

paper_set = set()
aurthor_list = list()
conference_set = set()
with open('dblp.conf20.csv', 'r') as f:
    for line in f:
        authors = line.split('\t')[0].split(',')
        year = int(line.split('\t')[2])
        # if len(authors)>1 and year >= min_year and year <= max_year:
        if year >= min_year and year <= max_year:
            paper_set.add(line.split('\t')[1])
            aurthor_list.extend(line.split('\t')[0].split(','))
            conference_set.add(line.split('\t')[3])
aurthor_list = list(set(aurthor_list))
paper_list = list(paper_set)
conference_list = list(conference_set)

aurthor_list.sort()
paper_list.sort()
conference_list.sort()
print("#author:%d #paper:%d #conf:%d" % (len(aurthor_list), len(paper_set), len(conference_set)))

line_num = 0
with open('dblp.conf20.csv', 'r') as f:
    for line in f:
        line_num += 1
print("line_num:%d" % line_num)
author_id = {author: i for i, author in enumerate(aurthor_list)}
conf_id = {conference: i + len(aurthor_list) for i, conference in enumerate(conference_set)}
paper_id = {paper: (i + len(aurthor_list) + len(conference_list)) for i, paper in enumerate(paper_list)}
id_dict = {**author_id, **conf_id}
# author:34312 #paper:27740 #conf:20
# line_num:29616

MMDNE = []
mp2v_edges = []

HHP_edges = []
nc_labels = {}  # node classification labels

lp_labels = {}  # link prediction labels

conf_author = {}
G = nx.Graph()  # 用于构建图，后面用于负采样

with open('dblp.conf20.csv', 'r') as f:
    for line in f:
        authors = line.split('\t')[0].split(',')
        paper = line.split('\t')[1]
        year = int(line.split('\t')[2])
        time = round(float(year - min_year) / dealt_year, 2)
        conf = line.split('\t')[3]

        if year == max_year:
            for author in authors:
                lp_labels[id_dict[author]] = id_dict[conf]
            for i in range(len(authors)):
                for j in range(len(authors)):
                    if i > j:
                        lp_labels[id_dict[authors[i]]] = id_dict[authors[j]]

        if paper in paper_id.keys() and year >= min_year and year <= max_year - 1:  # 节点分类数据集则需要把-1去掉
            for author in authors:
                mp2v_edges.append('\t'.join([str(id_dict[author]), str(id_dict[conf]), 'a', "c", "a-c", "1"]))
                mp2v_edges.append('\t'.join([str(id_dict[conf]), str(id_dict[author]), 'c', "a", "c-a", "1"]))

                HHP_edges.append(','.join([str(id_dict[author]), '0', str(id_dict[conf]), '1', '1', str(year)]))
                HHP_edges.append(','.join([str(id_dict[conf]), '1', str(id_dict[author]), '0', '2', str(year)]))

                MMDNE.append(str(id_dict[author]) + ' ' + str(id_dict[conf]) + ' ' + str(time))

                G.add_edge(id_dict[author], id_dict[conf])

            for author in authors:
                if paper in id_dict.keys() and author in id_dict.keys():
                    nc_labels[id_dict[author]] = confLabel[conf]  # 获得作者的标签
            #
            for i in range(len(authors)):
                for j in range(len(authors)):
                    if i > j:
                        MMDNE.append(str(id_dict[authors[i]]) + ' ' + str(id_dict[authors[j]]) + ' ' + str(time))

                        HHP_edges.append(
                            ','.join([str(id_dict[authors[i]]), '0', str(id_dict[authors[j]]), '0', '0', str(year)]))

                        G.add_edge(id_dict[authors[i]], id_dict[authors[j]])



with open("MM_dblp.txt", 'w') as f:
    for data in MMDNE:
        f.write(data + '\n')

with open("edge_c_dblp.txt", 'w') as f:
    for data in mp2v_edges:
        f.write(data + '\n')

with open("HHP_dblp.csv", 'w') as f:
    for data in HHP_edges:
        f.write(data + '\n')


with open('dblp_label_nc.pkl', 'wb') as f:
    pickle.dump(list(nc_labels.keys()), f)  # id
    pickle.dump(list(nc_labels.values()), f)  # 标签
    pickle.dump(4, f)  # 类的个数

snodes = []
tnodes = []
st_label = []

for snode, tnode in lp_labels.items():
    if snode in G.nodes and tnode in G.nodes:
        snodes.append(snode)
        tnodes.append(tnode)
        st_label.append(1)
        count = 0
        # 负采样
        while 1:  # 为snode采样负邻居 类型要和tnode一样
            if tnode in author_id.values():
                neg = random.sample(list(author_id.values()), 1)[0]

                if neg not in G[snode].keys():
                    snodes.append(snode)
                    tnodes.append(neg)
                    st_label.append(0)
                    count += 1
            else:
                neg = random.sample(list(conf_id.values()), 1)[0]

                if neg not in G[snode].keys():
                    count += 1
                    snodes.append(snode)
                    tnodes.append(neg)
                    st_label.append(0)
            if count == 2:
                break

        count = 0
        while 1:  # 为tnode采样负邻居 类型要和tnode一样
            if snode in author_id.values():
                neg = random.sample(list(author_id.values()), 1)[0]
                if neg not in G[tnode].keys():
                    count += 1
                    tnodes.append(tnode)
                    snodes.append(neg)
                    st_label.append(0)
            else:
                neg = random.sample(list(conf_id.values()), 1)[0]
                if neg not in G[tnode].keys():
                    count += 1
                    tnodes.append(tnode)
                    snodes.append(neg)
                    st_label.append(0)
            if count == 2:
                break

# 制作link prediction标签
with open('dblp_label_lp.pkl', 'wb') as f:
    pickle.dump(snodes,f)
    pickle.dump(tnodes,f)
    pickle.dump(st_label,f)
