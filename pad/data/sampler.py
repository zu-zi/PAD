"""Identity-balanced sampler for triplet-style training."""

import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """Sample ``num_instances`` images per identity to form each mini-batch."""

    def __init__(self, data_source, batch_size: int, num_instances: int):
        if batch_size < num_instances:
            raise ValueError("batch_size must be >= num_instances")
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            n = max(len(self.index_dic[pid]), self.num_instances)
            self.length += n - n % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected:
                final_idxs.extend(batch_idxs_dict[pid].pop(0))
                if not batch_idxs_dict[pid]:
                    avai_pids.remove(pid)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
