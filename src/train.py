import random
from pprint import pprint

import numpy as np
import termcolor
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import architectures
import argument_parser
import config
import datasets
import sampler
import math
import util as u
import warnings


def evaluate(model, test_loader, show_per_class_stats=False, persist=False, persist_suffix=None):
    model.eval()
    metric = u.Metric()
    total_len = len(test_loader)
    c = config.get_global_config()

    __feature_buffer = torch.empty(0)
    __logit_buffer = torch.empty(0)
    __path_buffer = np.empty(0)

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, start=1):
            print(f"\rEvaluating ({batch_idx:>5}/{total_len:>5}) ...{' ' * 100}", end='')
            label = data['label'].cuda(non_blocking=True).requires_grad_(False)
            img = data['image'].cuda().requires_grad_(False)
            o = edict(model(img, u.Phase(u.Phase.TEST)))
            metric.update_logits(o.logits, label)

            # Buffers
            __feature_buffer = torch.cat([__feature_buffer, o.features.detach().cpu()], dim=0)
            __logit_buffer = torch.cat([__logit_buffer, o.logits.detach().cpu()], dim=0)
            __path_buffer = np.hstack([__path_buffer, np.array(data['path'])])
    formatted_metric = termcolor.colored(f"{metric.get_accuracy() * 100:>6.3f}", on_color='on_grey')
    print(f"\r┏ Eval acc over {metric.get_element_count()} elements: {formatted_metric} {' ' * 100}")
    if show_per_class_stats:
        for k, v in metric.get_per_class_acc(test_loader.dataset.idx_to_class).items():
            print(f"┃ {k:<20}:   {v * 100:>6.3f}")
    # Print top-k acc
    output_buf = []
    for k in range(1, min(6, config.get_global_config().current.num_classes + 1)):
        output_buf.append(f"Top-{k} acc: {metric.get_top_k_accuracy(k) * 100:>6.3f}")
    print('┃', ' | '.join(output_buf))
    # Output mean class accuracy
    formatted_class_acc = termcolor.colored(f"{metric.get_mean_class_acc(test_loader.dataset.idx_to_class) * 100:.3f}", on_color='on_grey')
    print(f"┗ Mean over class accuracy: {formatted_class_acc}")

    # Persist results to filesystem
    if persist:
        with open(c.results_file.format(persist_suffix), 'a+') as f:
            f.write(f"{c.source_dataset},"
                    f"{c.target_dataset},"
                    f"{metric.get_accuracy() * 100},"
                    f"{metric.get_mean_class_acc(test_loader.dataset.idx_to_class) * 100}\n")
        torch.save({'paths': __path_buffer, 'features': __feature_buffer}, c.feature_file.format(persist_suffix))
    return __path_buffer, __feature_buffer, __logit_buffer


def train(model, loader, phase, iterations, optimizer, test_loader):
    c = config.get_global_config()
    metric = u.Metric()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.)

    for batch_idx, data in enumerate(loader, start=1):
        model.train()
        model.zero_grad()

        label = data['label'].cuda(non_blocking=True).requires_grad_(False)
        img = data['image'].cuda().requires_grad_(False)
        o = edict(model(img, phase))
        loss = u.softmax_crossentropy_with_logits(o.logits,
                                                  u.label_smoothing(label,
                                                                    eps=c.lsm_eps,
                                                                    n=c.current.num_classes,
                                                                    smooth_at=u.get_smoothing_positions(data['domain'], c.use_dss_for_pretraining)
                                                                    )
                                                  )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        metric.update_logits(o.logits, label)
        print(f"\r{batch_idx:>5}/{iterations:>5} ||| Accum. Acc: {metric.get_accuracy():.4f} (from {metric.get_element_count()} elements) {u.backfill()}",
              end='')

        if batch_idx % c.test_every == 0 and batch_idx > 0:
            del label, img
            evaluate(model, test_loader, persist=batch_idx == len(loader), persist_suffix=phase.get_phase(), show_per_class_stats=batch_idx == len(loader))
    # Flush to new line
    print('')


def extract_features(model, loader, repeat_from_config):
    model.eval()
    total_len = len(loader)

    output_buf = {}
    with torch.no_grad():
        for name, trans, model_phase, repeat in [('uncertainty',
                                                  u.Transforms.WEAK_UNCERTAINTY,
                                                  u.Phase(u.Phase.MC_DROPOUT),
                                                  repeat_from_config)]:
            output_buf[name] = {'paths': [np.empty(0)] * repeat, 'logits': [torch.empty(0)] * repeat, 'features': [torch.empty(0)] * repeat}
            for i in range(0, repeat):
                # Set data augmentations
                loader.dataset.set_transforms(trans)
                for batch_idx, data in enumerate(loader, start=1):
                    print(f"\r[{name.upper()}]: [{i + 1:>2}/{repeat:>2}] Extracting features [{batch_idx:>5}/{total_len:>5}] ...{u.backfill()}", end='')
                    img = data['image'].cuda().requires_grad_(False)
                    o = edict(model(img, model_phase))
                    output_buf[name]['features'][i] = torch.cat([output_buf[name]['features'][i], o.features.detach().cpu()], dim=0)
                    output_buf[name]['logits'][i] = torch.cat([output_buf[name]['logits'][i], o.mc_logits.detach().cpu()], dim=0)
                    output_buf[name]['paths'][i] = np.hstack([output_buf[name]['paths'][i], np.array(data['path'])])

            output_buf[name]['features'] = torch.cat([x.unsqueeze(1) for x in output_buf[name]['features']], dim=1)
            output_buf[name]['logits'] = torch.cat([x.unsqueeze(1) for x in output_buf[name]['logits']], dim=1)
            output_buf[name]['paths'] = np.concatenate([x.reshape(-1, 1) for x in output_buf[name]['paths']], axis=1)
    print(f"\rFeature extraction done (repeats={repeat}). {' ' * 100}")
    return output_buf


def adapt(model, loader, phase, optimizer, test_loader):
    c = config.get_global_config()
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.)

    # Augmented loader
    aug_tar_dataset = datasets.load_dataset(c.target_dataset, transforms=u.Transforms.UNCERTAINTY, keep_gt_label=False, domain=u.Domain.target())
    aug_tar_loader = DataLoader(aug_tar_dataset, batch_size=c.test_batch_size, shuffle=False, num_workers=c.workers, pin_memory=True, drop_last=False)

    for ada_cycle in range(1, c.adaptation_cycles + 1):
        print(termcolor.colored(f"\n[{phase.get_phase()}] cycle {ada_cycle}/{c.adaptation_cycles}: {c.task_dir}", on_color=f"on_{phase.get_color()}"))
        loader.batch_sampler.set_values(extract_features(model, aug_tar_loader, c.uncertain_repeats), domain=u.Domain.target())

        for batch_idx, data in enumerate(loader, start=1):
            model.train()
            label = data['label'].cuda(non_blocking=True).requires_grad_(False)
            img = data['image'].cuda().requires_grad_(False)
            o = edict(model(img, phase))
            loss = 0.

            # Uncertainty loss
            if c.use_decision_error or c.use_sample_likelihood:
                def gauss_cdf(x, mu, sigma):
                    return 0.5 * (1 + torch.erf((x - mu) / (sigma * math.sqrt(2))))

                target_indices = torch.nonzero(data['domain'] == u.Domain.target()).squeeze()
                target_current_labels = data['label'][target_indices]
                target_resampled_unc = data['resampled_uncertainty'][target_indices]
                target_unc_mean = data['uncertainty_mean'][target_indices]
                target_unc_std = data['uncertainty_std'][target_indices]
                # Liklihood of current sample
                _max_value = target_resampled_unc.gather(1, target_current_labels.view(-1, 1))
                _max_mean = target_unc_mean.gather(1, target_current_labels.view(-1, 1))
                _max_std = target_unc_std.gather(1, target_current_labels.view(-1, 1))
                normed_dist_to_mean = torch.abs(_max_value - _max_mean) / (2. * _max_std)
                sample_likelihood = torch.clamp(1. - normed_dist_to_mean, 0., 1.).squeeze()
                # Get estimate for how likely it is that another class is better
                resampled_value = target_resampled_unc.gather(1, target_current_labels.view(-1, 1)).view(-1, 1).expand_as(target_unc_mean)
                decision_error = (1. - gauss_cdf(resampled_value, target_unc_mean, target_unc_std))
                decision_error.scatter_(1, target_current_labels.view(-1, 1), torch.zeros_like(target_current_labels).view(-1, 1))
                decision_error = decision_error.max(dim=1).values.squeeze()
                if c.use_decision_error and c.use_sample_likelihood:
                    target_scores = (1. - decision_error) * sample_likelihood
                elif c.use_decision_error:
                    target_scores = (1. - decision_error)
                elif c.use_sample_likelihood:
                    target_scores = sample_likelihood
                else:
                    raise AttributeError()
                loss_weights = torch.ones(c.batch_size)
                loss_weights[target_indices] = (target_scores / target_scores.mean())

            # Xent loss with label smoothing, get positions to smooth at
            loss += u.softmax_crossentropy_with_logits(o.logits,
                                                       u.label_smoothing(label,
                                                                         eps=c.lsm_eps,
                                                                         n=c.current.num_classes,
                                                                         smooth_at=u.get_smoothing_positions(data['domain'], c.use_dss_for)),
                                                       weight=loss_weights if (c.use_decision_error or c.use_sample_likelihood) else None
                                                       )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            print(f"\rAdapting [{batch_idx:>4}/{len(loader):>4}]: "
                  f"BS {img.shape[0]}, "
                  f"LR {optimizer.param_groups[0]['lr']}, "
                  f"Tar%: {(data['domain'] == u.Domain.target()).float().sum() / len(data['domain']) * 100:.1f}, "
                  f"Loss: {loss.item():.3f}, "
                  f"{u.backfill()}",
                  end='')
        # Reset cursor
        print('')
        if ada_cycle % c.eval_every_adapt_cycle == 0 and ada_cycle != c.adaptation_cycles:
            evaluate(model, test_loader, show_per_class_stats=False)


def get_loaders():
    c = config.get_global_config()
    # Load datasets
    source_dataset = datasets.load_dataset(c.source_dataset, transforms=u.Transforms.TRAIN, domain=u.Domain.source(), keep_gt_label=True)
    source_loader = DataLoader(source_dataset, num_workers=c.workers, pin_memory=True, batch_sampler=sampler.EndlessBatchSampler(source_dataset))
    evaluation_dataset = datasets.load_dataset(c.target_dataset, transforms=u.Transforms.TEST, domain=u.Domain.target(), keep_gt_label=True)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True, drop_last=False)
    return source_dataset, source_loader, evaluation_dataset, evaluation_loader


def main():
    # Fix for PyTorch warning that is not actually correct
    warnings.filterwarnings('ignore', category=UserWarning)

    # Load config and set initial seeds for reproducibility
    argument_parser.parse()
    c = config.get_global_config()
    random.seed(c.seed)
    torch.manual_seed(c.seed)
    torch.cuda.manual_seed(c.seed)
    torch.cuda.manual_seed_all(c.seed)
    np.random.seed(c.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Print current config
    pprint(config.get_minimal_config())

    model = architectures.get_model(c.architecture, c.gpus)
    # If snapshot is set, overwrite current_phase and restore weights and RNG states
    if c.snapshot is not None:
        loaded_data, loaded_config, current_phase = u.load_snapshot(c.snapshot, model)
    else:
        current_phase = u.Phase(u.Phase.SOURCE_ONLY_TRAIN)

    # Training phases
    if current_phase == u.Phase(u.Phase.SOURCE_ONLY_TRAIN):
        optimizer = u.get_optimizer(model, current_phase)
        source_dataset, source_loader, evaluation_dataset, evaluation_loader = get_loaders()
        with u.AutoSnapshotter(model, optimizer, current_phase, next_phase=u.Phase(u.Phase.ADAPTATION_TRAIN)):
            # Train on source only
            train(model, source_loader,
                  phase=current_phase,
                  iterations=c.source_iterations,
                  optimizer=optimizer,
                  test_loader=evaluation_loader)

    if current_phase == u.Phase(u.Phase.ADAPTATION_TRAIN):
        optimizer = u.get_optimizer(model, current_phase)
        source_dataset, _, _, evaluation_loader = get_loaders()
        with u.AutoSnapshotter(model, optimizer, current_phase, next_phase=u.Phase(u.Phase.FINAL_TEST)):
            # Adapt model on target
            target_dataset = datasets.load_dataset(c.target_dataset, transforms=u.Transforms.TRAIN, keep_gt_label=False, domain=u.Domain.target())
            concat_dataset = datasets.ConcatDataset(source=source_dataset, target=target_dataset)
            # noinspection PyArgumentList
            concat_loader = DataLoader(concat_dataset,
                                       batch_sampler=sampler.BatchSampler(concat_dataset),
                                       num_workers=c.workers,
                                       pin_memory=True)
            # noinspection PyTypeChecker
            adapt(model, concat_loader,
                  phase=current_phase,
                  optimizer=optimizer,
                  test_loader=evaluation_loader)

    if current_phase == u.Phase(u.Phase.FINAL_TEST):
        _, _, _, evaluation_loader = get_loaders()
        with u.AutoSnapshotter(model, None, current_phase, next_phase=u.Phase(u.Phase.FINAL_TEST)):
            evaluate(model, evaluation_loader, show_per_class_stats=True, persist=True, persist_suffix=current_phase.get_phase())


if __name__ == '__main__':
    main()
