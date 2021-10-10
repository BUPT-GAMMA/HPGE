from tqdm import tqdm

from tools import *

NEG_SAMPLING_POWER = 0.75


class DataLoader:

    def __init__(self, path, filename, train_test=-1, nbr_size=1, neg_size=1, delim='', num_edge_types=4,
                 sample_type='cut_off'):
        # *** graph should contains A-B and B-A
        self.graph = {}
        self.nodes = {}
        self.num_edge_types = num_edge_types
        self.path = path

        self.nbr_size = nbr_size
        self.neg_size = neg_size

        self.id_map_sid = []
        self.id_map_s_type = []
        self.id_map_tid = []
        self.id_map_t_type = []
        self.id_map_e_type = []
        self.id_map_time = []
        self.max_d_time = 0
        self.node_ids_dict = {}
        self.sample_type = sample_type
        with open(filename) as rf:
            while 1:
                line = rf.readline()
                if not line:
                    break
                sid, s_type, tid, t_type, e_type, timestamp = [int(float(x)) for x in line.strip().split(delim)]
                # TODO timestamp should be normalized
                self.nodes.setdefault(s_type, {})
                self.nodes[s_type].setdefault(sid, 0)
                self.nodes[s_type][sid] += 1
                self.node_ids_dict.setdefault(sid, 1)
                self.node_ids_dict.setdefault(tid, 1)

                self.max_d_time = max(self.max_d_time, timestamp)

                if timestamp <= train_test or train_test == -1:
                    temp = self.graph.setdefault(sid, {})
                    temp.setdefault(e_type, []).append([tid, timestamp])
                    self.id_map_sid.append(sid)
                    self.id_map_s_type.append(s_type)
                    self.id_map_tid.append(tid)
                    self.id_map_t_type.append(t_type)
                    self.id_map_e_type.append(e_type)
                    self.id_map_time.append(timestamp)

        self.data_size = len(self.id_map_e_type)
        self.node_size = len(self.node_ids_dict.keys())

        self.min_d_time = min(self.id_map_time)
        print(self.max_d_time, self.min_d_time)
        self.timespan = (self.max_d_time - self.min_d_time)

        # sort based on timestamp
        for nid, hete_edges in self.graph.items():
            for e_type, e_list in hete_edges.items():
                hete_edges[e_type] = sorted(e_list, key=lambda x: x[-1])
            self.graph[nid] = hete_edges
        print("The dataset have been loaded. \n Edges: {}, Nodes: {}".format(self.data_size, self.node_size))

        # init negative table
        self.neg_table = {}
        for n_type in self.nodes.keys():
            self.neg_table[n_type] = self.init_neg_table(n_type)

        self.node_size_type = {}
        for n_type, node_list in self.nodes.items():
            self.node_size_type[n_type] = len(node_list)

    def init_neg_table(self, type):
        nodes = self.nodes[type]
        node_ids = list(nodes.keys())
        node_degrees = list(nodes.values())
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        tot_sum = np.power(node_degrees, NEG_SAMPLING_POWER).sum()
        node_size = len(nodes)
        neg_table = np.zeros(node_size, )
        for k in range(node_size):
            if (k + 1.) / node_size > por:
                cur_sum += np.power(node_degrees[n_id], NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            neg_table[k] = node_ids[n_id - 1]
        return neg_table

    def negative_sampling(self, n_type):
        rand_idx = np.random.randint(0, self.node_size_type[n_type], (self.neg_size,))
        sampled_nodes = self.neg_table[n_type][rand_idx]
        sampled_nodes = ",".join(np.array(sampled_nodes, dtype=np.int).astype(np.str))
        return sampled_nodes

    def generate_whole_node_nbrs(self):
        node_nbrs = {}
        process_node = tqdm(total=len(self.graph))
        count = 0
        for nid, hete_nbrs in self.graph.items():
            count += 1
            if count % 100 == 0:
                process_node.update(100)
            temp = {}
            for e_type, e_list in hete_nbrs.items():
                temp[e_type] = {}
                sampled_nbrs = self.node_neighbor_sampling(np.array(e_list), e_list[-1][1])
                for i, [ids, weights, timestamp] in enumerate(sampled_nbrs):
                    temp[e_type][timestamp] = [ids, weights]
            node_nbrs[nid] = temp
        process_node.close()
        return node_nbrs

    def generate_training_dataset(self, filename, num_process=10):
        node_nbrs = self.generate_whole_node_nbrs()
        with open(filename, "w") as wf:
            process = tqdm(total=len(self.id_map_e_type))
            for i in range(len(self.id_map_e_type)):
                if (i + 1) % 10000 == 0:
                    process.update(10000)
                sid = self.id_map_sid[i]
                s_type = self.id_map_s_type[i]
                tid = self.id_map_tid[i]
                t_type = self.id_map_t_type[i]
                e_type = self.id_map_e_type[i]
                timestamp = self.id_map_time[i]

                neg_s_nodes = self.negative_sampling(s_type)
                neg_t_nodes = self.negative_sampling(t_type)
                s_hist_ids = ['' for _ in range(self.num_edge_types)]
                s_hist_weights = ['' for _ in range(self.num_edge_types)]
                s_hist_flags = ['-1' for _ in range(self.num_edge_types)]
                for et, sampled_nbrs in node_nbrs[sid].items():
                    if timestamp in sampled_nbrs:
                        temp_ids, temp_weights = sampled_nbrs[timestamp]
                        s_hist_ids[et] = temp_ids
                        s_hist_weights[et] = temp_weights
                        s_hist_flags[et] = '1'
                t_hist_ids = ['' for _ in range(self.num_edge_types)]
                t_hist_weights = ['' for _ in range(self.num_edge_types)]
                t_hist_flags = ['-1' for _ in range(self.num_edge_types)]
                for et, sampled_nbrs in node_nbrs[tid].items():
                    if timestamp in sampled_nbrs:
                        temp_ids, temp_weights = sampled_nbrs[timestamp]
                        t_hist_ids[et] = temp_ids
                        t_hist_weights[et] = temp_weights
                        t_hist_flags[et] = '1'
                outs = [str(e_type), str(sid), str(s_type), neg_s_nodes] + s_hist_ids + s_hist_weights + s_hist_flags + \
                       [str(tid), str(t_type), neg_t_nodes] + t_hist_ids + t_hist_weights + t_hist_flags
                train_info = ";".join(outs) + "\n"
                wf.write(train_info)
            process.close()
        return train_info

    def node_neighbor_sampling(self, node_nbrs, t):
        if len(node_nbrs) == 0:
            return []
        else:
            times = node_nbrs[:, 1]
            ids = node_nbrs[:, 0]
            delta_t = (times - t) * 1.0 / self.timespan
            p = np.exp(delta_t)
            outs = self.importance_sampler(ids, p) if self.sample_type == 'important' else self.cutoff_sampler(ids, p)
            outs.append(t)
            new_t = node_nbrs[-1][1]
            new_node_nbr_idx = node_nbrs[(np.where(node_nbrs[:, 1] < new_t))]
            return [outs] + self.node_neighbor_sampling(new_node_nbr_idx, new_t)

    def importance_sampler(self, ids, p):
        uniq_ids, ids_index, ids_inverse = np.unique(ids, return_index=True, return_inverse=True)
        id_matrix = np.eye(len(uniq_ids), dtype=np.int)[ids_inverse]
        sum_uniq_p = np.dot(p, id_matrix).reshape(-1)  # 1 * d
        sum_uniq_q = sum_uniq_p ** 2
        norm_q = sum_uniq_q / np.sum(sum_uniq_q)
        sampled_ids = np.random.choice(np.arange(len(uniq_ids)), size=self.nbr_size, p=norm_q, replace=True)
        sp_ids, sp_counts = np.unique(sampled_ids, return_counts=True)
        weight = np.multiply((sum_uniq_p / norm_q)[sp_ids], sp_counts * 1.0 / self.nbr_size)
        norm_weight = weight / weight.sum()
        sp_node_ids = uniq_ids[sp_ids]
        return [','.join(sp_node_ids.astype(np.str)), ','.join(norm_weight.astype(np.str))]

    def cutoff_sampler(self, ids, p):
        if self.nbr_size == 0:
            return ['', '']
        elif len(ids) < self.nbr_size:
            return [','.join(ids.astype(np.str)), ','.join(np.array(p).astype(np.str))]
        else:
            return [','.join(ids.astype(np.str)[len(ids) - self.nbr_size:]),
                    ','.join(np.array(p).astype(np.str)[len(ids) - self.nbr_size:])]
#
# if __name__ == "__main__":
#     # path = "/home/jiyugang/work2020/HHP/dataset/aminer/"
#     path = "/home/jyg/work2020/HHP/dataset/aminer/"
#     filename = path + "graph.csv"
#     delim = ","
#     data_loader = DataLoader(path, filename, delim=delim, nbr_size=5, neg_size=5)
#     data_loader.generate_training_dataset('/home/jyg/work2020/HHP/dataset/aminer/train_test_{}_{}', 10)
