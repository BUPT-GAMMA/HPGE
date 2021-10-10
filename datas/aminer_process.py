import pickle
import random
import networkx as nx

filenames = ['Database.txt', 'DataMining.txt', 'MedicalInformatics.txt', 'Theory.txt', 'Visualization.txt']
author_set = set()
conf_set = set()
max_year = -1000000
min_year = 1000000

# 获得author_set
cate_dict = {}
for i, name in enumerate(filenames):
    with open('Cross_Domain_data/' + name, 'r') as f:
        for line in f.readlines():
            author_string = line.split('\t')[2]
            authors = author_string.split(',')
            conf_set.add(line.split('\t')[0])

            for author in authors:
                if author not in cate_dict.keys():
                    cate_dict[author] = i
                author_set.add(author)
            # 获得年份最大值最小值
            year = int(line.split('\t')[3])
            if year > max_year:
                max_year = year
            if year < min_year:
                min_year = year
print("最大年份：", max_year)
print("最小年份：", min_year)

dealt_year = max_year - min_year

G = nx.Graph()  # 用于构建图，后面用于负采样

# 将集合转化为字典 自动编号
author_id = {}
conf_id = {}

ids = {}
author_list = list(author_set)
author_list.sort()
conf_list = list(conf_set)
conf_list.sort()
test = []

for i, name in enumerate(author_list):
    ids[name] = str(i)
    author_id[name] = str(i)
start_id = len(ids)

for i, conf in enumerate(conf_list):
    ids[conf] = str(i + start_id)
    conf_id[conf] = str(i + start_id)
edgelist = []
lp_labels = {}

for i, name in enumerate(filenames):

    with open('Cross_Domain_data/' + name, 'r') as f:
        for line in f.readlines():
            # 获得作者
            author_string = line.split('\t')[2]
            authors = author_string.split(',')
            # 获得年份
            year = float(line.split('\t')[3])
            time = round((year - min_year) / dealt_year, 5)
            conf = line.split('\t')[0]
            if year != max_year:

                for i in range(len(authors)):
                    for j in range(len(authors)):
                        if i < j:
                            G.add_edge(ids[authors[i]], ids[authors[j]])
                            edgelist.append(ids[authors[i]] + ',0,' + ids[authors[j]] + ',0,0,' + str(year))
                    G.add_edge(ids[authors[i]], ids[conf])
                    edgelist.append(ids[authors[i]] + ',0,' + ids[conf] + ',1,1,' + str(year))
                    edgelist.append(ids[conf] + ',1,' + ids[authors[i]] + ',0,2,' + str(year))
            else:
                for author in authors:
                    lp_labels[ids[author]] = ids[conf]
                for i in range(len(authors)):
                    for j in range(len(authors)):
                        if i > j:
                            lp_labels[ids[authors[i]]] = ids[authors[j]]


# a:0 c:1 aa:0 ac:1 ca:2

with open('aminer_graph.csv', 'w') as wf:
    for data in edgelist:
        wf.write(data + '\n')

# MAKE LABEL FILE
# with open('aminer_label.txt','w') as wf:
#     for i in range(len(author_list)):
#         wf.write(str(ids[author_list[i]])+' '+str(cate_dict[author_list[i]])+'\n')

label_author = []
label_cate = []
for each in author_list:
    label_author.append(int(ids[each]))
    label_cate.append(cate_dict[each])
# 用pkl文件写入标签
with open('aminer_label_nc.pkl','wb') as f:
    pickle.dump(label_author,f)
    pickle.dump(label_cate,f)
    pickle.dump(len(cate_dict), f)

snodes = []
tnodes = []
st_label = []
for snode, tnode in lp_labels.items():
    if snode in G.nodes and tnode in G.nodes:
        snodes.append(int(snode))
        tnodes.append(int(tnode))
        st_label.append(1)
        count = 0
        # 负采样
        while 1:  # 为snode采样负邻居 类型要和tnode一样
            if tnode in author_id.values():
                neg = random.sample(list(author_id.values()), 1)[0]
                if neg not in G[snode].keys():
                    snodes.append(int(snode))
                    tnodes.append(int(neg))
                    st_label.append(0)
                    count += 1
            else:
                neg = random.sample(list(conf_id.values()), 1)[0]

                if neg not in G[snode].keys():
                    snodes.append(int(snode))
                    tnodes.append(int(neg))
                    st_label.append(0)
                    count += 1
            if count == 2:
                break

        count = 0
        while 1:  # 为tnode采样负邻居 类型要和tnode一样
            if snode in author_id.values():
                neg = random.sample(list(author_id.values()), 1)[0]
                if neg not in G[tnode].keys():
                    count += 1
                    tnodes.append(int(tnode))
                    snodes.append(int(neg))
                    st_label.append(0)
            else:
                neg = random.sample(list(conf_id.values()), 1)[0]
                if neg not in G[tnode].keys():
                    count += 1
                    tnodes.append(int(tnode))
                    snodes.append(int(neg))
                    st_label.append(0)
            if count == 2:
                break

# 制作link prediction标签
with open('aminer_label_lp.pkl', 'wb') as f:
    pickle.dump(snodes,f)
    pickle.dump(tnodes,f)
    pickle.dump(st_label,f)
