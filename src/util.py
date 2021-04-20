import socket
import datetime
import os
import os.path as osp
import shutil
import config
import json
import torch
import torch.optim
import numpy as np
import termcolor
import random
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image


def print_header(phase):
    c = config.get_global_config()
    print(termcolor.colored(f"\n\n\nEntering phase [{phase.get_phase()}]: {c.task_dir}", on_color=f"on_{phase.get_color()}"))


def save_setup():
    c = config.get_global_config()
    os.makedirs(c['task_dir'], exist_ok=True)

    with open('/proc/self/cmdline', 'r') as f:
        cmd_line = f.read().replace('\x00', ' ').strip()
    with open(osp.join(c['task_dir'], 'cmd.sh'), 'w+') as f:
        f.write(cmd_line)
    with open(osp.join(c['task_dir'], 'info.json'), 'w+') as f:
        f.write(json.dumps({'host': socket.gethostname(), 'date': str(datetime.datetime.now())}, indent=4))

    shutil.copytree('./src', osp.join(c['task_dir'], 'src'))
    shutil.copytree('./configs', osp.join(c['task_dir'], 'configs'))


def save_snapshot(model, optimizer, phase, extra_data=None):
    c = config.get_global_config()
    save_dict = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict() if optimizer is not None else None,
                 'RNG': {
                     'python': random.getstate(),
                     'numpy': np.random.get_state(),
                     'pytorch': torch.random.get_rng_state(),
                     'cuda': torch.cuda.get_rng_state_all(),
                     'seed': c.seed
                 },
                 'data': extra_data,
                 'config': c.__dict__,
                 'active_phase': phase.get_phase(),
                 }
    torch.save(save_dict, c.snapshot_path)


class AutoSnapshotter:
    def __init__(self, model, optimizer, current_phase, next_phase, extra_data=None):
        self.model = model
        self.optimizer = optimizer
        self.current_phase = current_phase
        self.next_phase = next_phase
        self.extra_data = extra_data

    def __enter__(self):
        print_header(self.current_phase)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Do not snapshot if error occurs within the context
        if exc_type is None:
            save_snapshot(self.model, self.optimizer, self.next_phase, self.extra_data)
            self.current_phase.inplace_update(self.next_phase.get_phase())


def load_snapshot(path, model):
    print(f"Restoring snapshot from {path}!")
    c = config.get_global_config()
    save_dict = torch.load(path)
    assert c.seed == save_dict['RNG']['seed'], f"Current seed {c.seed} does not match snapshot seed {save_dict['RNG']['seed']}."
    model.load_state_dict(save_dict['model'])
    random.setstate(save_dict['RNG']['python'])
    torch.random.set_rng_state(save_dict['RNG']['pytorch'])
    torch.cuda.set_rng_state_all(save_dict['RNG']['cuda'])
    np.random.set_state(save_dict['RNG']['numpy'])
    return save_dict['data'], save_dict['config'], Phase(save_dict['active_phase'])


def get_optimizer(m, phase):
    c = config.get_global_config()
    params = m.module.get_parameters()
    return getattr(torch.optim, c.optimizer)(params, **c.optimizer_params[phase.get_phase()])


class Domain:
    SOURCE = 'SOURCE'
    TARGET = 'TARGET'

    def __init__(self, name):
        assert name in [self.SOURCE, self.TARGET]
        self.domain = name

    def as_int(self):
        return {self.SOURCE: 0, self.TARGET: 1}[self.domain]

    @staticmethod
    def source():
        return Domain(Domain.SOURCE)

    @staticmethod
    def target():
        return Domain(Domain.TARGET)

    def get_text(self):
        return self.domain

    def __eq__(self, other):
        if isinstance(other, str):
            return self.domain == other
        elif isinstance(other, (int, float)):
            return self.as_int() == other
        elif isinstance(other, Domain):
            return self.domain == other.domain
        elif isinstance(other, list):
            return [self.__eq__(x) for x in other]
        elif isinstance(other, torch.Tensor):
            return torch.tensor([self.__eq__(x.item()) for x in other]).bool()
        elif isinstance(other, np.ndarray):
            return np.array([self.__eq__(x.item()) for x in other], dtype=np.bool)
        else:
            raise ValueError

    def __str__(self):
        return f"<Domain {self.domain}>"


class Transforms:
    TRAIN = TF.Compose([
        TF.RandomResizedCrop((256, 256), scale=(0.75, 1.25), interpolation=Image.BICUBIC),
        TF.RandomHorizontalFlip(p=0.5),
        TF.RandomAffine(10, translate=(0.025, 0.025), scale=(0.975, 1.025), shear=10, resample=Image.BICUBIC),
        TF.RandomChoice([
            TF.RandomGrayscale(p=0.2),
            TF.ColorJitter(brightness=0.05, saturation=0.05, contrast=0.05, hue=0.05)
        ]),
        TF.RandomCrop((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    TEST = TF.Compose([
        TF.Resize((256, 256)),
        TF.CenterCrop((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    UNCERTAINTY = TF.Compose([
        TF.RandomResizedCrop((256, 256), scale=(0.75, 1.25), interpolation=Image.BICUBIC),
        TF.RandomHorizontalFlip(p=0.5),
        TF.RandomAffine(10, translate=(0.025, 0.025), scale=(0.975, 1.025), shear=10, resample=Image.BICUBIC),
        TF.RandomChoice([
            TF.RandomGrayscale(p=0.2),
            TF.ColorJitter(brightness=0.05, saturation=0.05, contrast=0.05, hue=0.05)
        ]),
        TF.RandomCrop((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    WEAK_UNCERTAINTY = TF.Compose([
        TF.RandomResizedCrop((256, 256), scale=(0.75, 1.25), interpolation=Image.BICUBIC),
        TF.RandomHorizontalFlip(p=0.5),
        TF.RandomAffine(10, translate=(0.025, 0.025), scale=(0.975, 1.025), shear=10, resample=Image.BICUBIC),
        TF.RandomCrop((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


class Phase:
    TRAIN = 'TRAIN'
    TEST = 'TEST'
    ADAPTATION_TRAIN = 'ADAPTATION_TRAIN'
    SOURCE_ONLY_TRAIN = 'SOURCE_ONLY_TRAIN'
    FINAL_TEST = 'FINAL_TEST'
    FEATURE_EXTRACTION = 'FEATURE_EXTRACTION'
    MC_DROPOUT = 'MC_DROPOUT'

    def __init__(self, phase_name: str):
        assert phase_name.upper() in Phase.__dict__
        self.phase = phase_name.upper()
        self._phases = {Phase.TRAIN: 0,
                        Phase.TEST: 2,
                        Phase.ADAPTATION_TRAIN: 4,
                        Phase.FINAL_TEST: 6,
                        Phase.FEATURE_EXTRACTION: 10,
                        Phase.MC_DROPOUT: 11}

    def inplace_update(self, new_phase):
        assert new_phase.upper() in Phase.__dict__
        self.phase = new_phase

    def as_int(self) -> int:
        return self._phases[self.phase]

    def is_train(self):
        return self.phase in [self.TRAIN, self.ADAPTATION_TRAIN, self.SOURCE_ONLY_TRAIN]

    def is_test(self):
        return self.phase in [self.TEST, self.FINAL_TEST, self.FEATURE_EXTRACTION, self.MC_DROPOUT]

    def get_phase(self):
        return self.phase

    def __eq__(self, other):
        return self.phase == other.phase

    def get_color(self):
        return {self.TRAIN: None,
                self.TEST: 'cyan',
                self.ADAPTATION_TRAIN: 'blue',
                self.FINAL_TEST: 'magenta',
                self.FEATURE_EXTRACTION: 'magenta'}.get(self.phase, 'grey')


class Metric:
    def __init__(self):
        self.logit_buf = None
        self.target_buf = None
        self.reset()

    def reset(self):
        self.logit_buf = torch.empty(0)
        self.target_buf = torch.empty(0).long()

    def update_logits(self, logits, target):
        logits = logits.cpu().detach()
        target = target.cpu().detach()
        self.logit_buf = torch.cat([self.logit_buf, logits], dim=0)
        self.target_buf = torch.cat([self.target_buf, target.view(-1, 1)], dim=0)

    def get_accuracy(self):
        return float((self.logit_buf.argmax(dim=1) == self.target_buf.squeeze()).float().sum() / self.target_buf.shape[0])

    def get_top_k_accuracy(self, k=1):
        top_k_indices = self.logit_buf.topk(dim=1, k=k).indices
        exp_targets = self.target_buf.expand_as(top_k_indices)
        correct_preds = (top_k_indices == exp_targets).float().sum(dim=1).clamp(0, 1)
        top_k_acc = float(correct_preds.sum() / correct_preds.shape[0])
        return top_k_acc

    def get_element_count(self):
        assert self.logit_buf.shape[0] == self.target_buf.shape[0]
        return self.target_buf.shape[0]

    def get_per_class_acc(self, mapping):
        accs = {}
        sorted_items = sorted(list(mapping.items()), key=lambda x: x[1])
        assert len(sorted_items) == config.get_global_config().current.num_classes
        ctr = 0
        for class_idx, class_name in sorted_items:
            indicies = (self.target_buf.squeeze() == class_idx)
            ctr += int(indicies.float().sum())
            accs[class_name] = float((self.logit_buf[indicies].argmax(dim=1) == self.target_buf[indicies].squeeze()).float().sum() / indicies.float().sum())
        assert len(accs.keys()) == config.get_global_config().current.num_classes
        assert ctr == self.get_element_count()
        return accs

    def get_mean_class_acc(self, mapping):
        return np.mean(list(self.get_per_class_acc(mapping).values()))


def softmax_crossentropy_with_logits(logits, target, reduction='mean', weight=None):
    """ TF-like crossentropy implementation, as PyTorch only accepts the class index and not a one hot encoded label.
    :param logits: NxC matrix of logits.
    :param target: NxC matrix of probability distribution.
    :return: Average loss over all batch elements.
    """
    loss = torch.sum(- target * F.log_softmax(logits, -1), -1)
    if weight is not None:
        loss *= weight.cuda()
    if reduction == 'mean':
        loss = loss.mean()
    return loss


def label_smoothing(x, eps, n, smooth_at=None):
    num_samples = x.shape[0]
    smoothed_target = torch.zeros(num_samples, n).fill_(eps / (n - 1)).cuda()
    smoothed_target.scatter_(1, x.unsqueeze(1), 1. - eps)

    if smooth_at is not None:
        hard_target = torch.zeros(num_samples, n).cuda().scatter_(1, x.unsqueeze(1), 1.)
        indices = ~(smooth_at.bool())
        smoothed_target[indices] = hard_target[indices]
    return smoothed_target


def sample_avoid_dupes(array, num: int):
    num = int(num)
    if num == 0:
        return []
    elif len(array) >= num:
        return np.random.choice(array, size=num, replace=False).tolist()
    else:
        buffer = []
        while len(buffer) != num:
            buffer.extend(np.random.choice(array, size=min([len(array), num - len(buffer)]), replace=False).tolist())
            assert len(buffer) <= num, (len(buffer), num)
        return buffer


def backfill():
    return f"{chr(9) * 200}"


def to_list(x: torch.Tensor):
    z = x.cpu().detach().numpy().tolist()
    if isinstance(z, list):
        return z
    else:
        return [z]


def get_smoothing_positions(domain_flag, domains: list):
    smoothing_positions = torch.zeros_like(domain_flag).bool()
    if domains is not None and len(domains) > 0:
        for d in domains:
            smoothing_positions |= (domain_flag == Domain(d))
    return smoothing_positions
