from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler as PTBatchSampler
import math
import config
import util as u
from easydict import EasyDict as edict
import random
from functools import partial
import copy


class BatchSampler(PTBatchSampler):
    def __init__(self, concat_dataset):
        self.c = config.get_global_config()
        self.concat_dataset = concat_dataset
        self.multi_source_ptr = 0

        # Uncertain results
        self.o = None

    def __len__(self):
        return self.c.adaptation_iterations

    def __iter__(self):
        c = config.get_global_config()

        # Create source structure
        source_structure = defaultdict(partial(defaultdict, list))
        for inst in self.concat_dataset.source_dataset:
            source_structure[inst.dataset_subdomain][inst.class_index].append(inst)

        # Prepare uncertainty
        softmaxed_logits = self.o.uncertainty['logits'].softmax(dim=-1)
        mean_probs = softmaxed_logits.mean(dim=1).mean(dim=1)
        std_probs = softmaxed_logits.mean(dim=1).std(dim=1)
        for resample_iteration in range(c.adaptation_iterations // c.adaptation_resample_every):
            # Create target structure for current iteration
            target_structure = defaultdict(list)
            # Resample probs
            resampled_probs = torch.clamp(mean_probs + std_probs * torch.randn_like(std_probs), 0., 1.)
            renormed_resampled_probs = resampled_probs / resampled_probs.sum(dim=1, keepdim=True)
            # Technically, there can be zero divisions so check for NaN values
            nan_flag = torch.any(torch.isnan(renormed_resampled_probs), dim=1).squeeze()

            for idx, inst in enumerate(self.concat_dataset.get_target_domain()):
                if not nan_flag[idx]:
                    weights = renormed_resampled_probs[idx].squeeze().cpu().detach().numpy().astype(np.float32)
                    chosen_idx = random.choices(range(len(weights)), weights=weights, k=1)[0]

                    # Inst is now class <chosen_idx>
                    assert not inst.tainted
                    assert inst.class_index is None
                    _inst = copy.deepcopy(inst)
                    _inst.class_index = int(chosen_idx)
                    _inst.softmax_prob = float(weights[chosen_idx])
                    _inst.uncertainty_mean = mean_probs[idx]
                    _inst.uncertainty_std = std_probs[idx]
                    _inst.resampled_uncertainty = resampled_probs[idx]
                    _inst.tainted = True
                    target_structure[int(mean_probs[idx].argmax())].append(_inst)

            # Sort in descending order
            for k, v in target_structure.items():
                target_structure[k] = list(sorted(v, key=lambda x: x.softmax_prob, reverse=True))

            yield from self.sample_batch(source_structure, target_structure, num_batches=c.adaptation_resample_every)

    def sample_batch(self, s_struct, t_struct, num_batches):
        c = config.get_global_config()

        # Actually sample and interleave batch
        for _ in range(num_batches):
            # Choose a subdomain to sample from in this iteration, this does only matter for multi-source DA
            if len(s_struct.keys()) > 1:
                sampled_d_subdomain = list(sorted(s_struct.keys()))[self.multi_source_ptr]
                self.multi_source_ptr = (self.multi_source_ptr + 1) % len(s_struct.keys())
            else:
                sampled_d_subdomain = list(s_struct.keys())[0]

            current_batch = []
            sampled_classes = u.sample_avoid_dupes(np.arange(c.current.num_classes, dtype=np.int32), c.sample_num_classes)
            per_domain_samples = c.batch_size / (2 * c.sample_num_classes)

            for cl in sampled_classes:
                tmp_source, tmp_target = [], []
                # Sample source
                tmp_source.extend(u.sample_avoid_dupes(s_struct[sampled_d_subdomain][cl], per_domain_samples))
                # Sample target
                if cl in t_struct and len(t_struct[cl]) >= len(c.sample_bins):
                    cnt = len(t_struct[cl])
                    num_bins = len(c.sample_bins)
                    for i, num in enumerate(c.sample_bins):
                        tmp_target.extend(u.sample_avoid_dupes(t_struct[cl][int(i * cnt / num_bins): int((i + 1) * cnt / num_bins)], num))
                else:
                    tmp_target.extend(u.sample_avoid_dupes(s_struct[sampled_d_subdomain][cl], per_domain_samples))

                # Interleave
                for s, t in zip(tmp_source, tmp_target):
                    current_batch.extend((s, t))
            yield current_batch

    def set_values(self, output_buf, domain):
        # features, logits, paths, certain_logits, certain_features,
        if domain == u.Domain.target():
            self.o = edict(output_buf)


class EndlessBatchSampler(PTBatchSampler):
    """Fixes stupid PyTorch dataloader behavior for small datasets.
    """

    def __init__(self, d):
        self.c = config.get_global_config()
        self.d = d

    def __len__(self):
        return self.c.source_iterations

    def __iter__(self):
        batch_size = self.c.batch_size
        ind_buf = []
        while len(ind_buf) < len(self) * batch_size:
            ind_buf.extend(torch.randperm(len(self.d)).tolist())

        # Sample one batch
        for batch_idx in range(len(self)):
            yield ind_buf[batch_idx * batch_size: (batch_idx + 1) * batch_size]
