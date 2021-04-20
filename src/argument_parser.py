import datetime
import socket
import argparse
import datasets
import config
import os
import util as u
import os.path as osp


def remove_extension(s, ext=('yaml',)):
    for ex in ext:
        if s.endswith(ex):
            return s[:-len(ex) - 1]
    return s


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dataset', type=str, choices=datasets.get_available(), nargs='+')
    parser.add_argument('--target-dataset', type=str, choices=datasets.get_available(), nargs='+')
    parser.add_argument('--configs', type=str, default=[], nargs='+')
    parser.add_argument('--sub-dir', type=str, default='')
    parser.add_argument('--task-dir', type=str, default=None)
    parser.add_argument('--exp-dir', type=str, default=None)
    parser.add_argument('--snapshot', type=str, default=None)
    args = parser.parse_args()
    config.set_cli_args(args)
    for sub_config in args.configs:
        config.merge_into_global_from_file(sub_config)
    c = config.get_global_config()

    source_datasets = [x.split(datasets.DATASET_SEPARATOR)[0] for x in args.source_dataset]
    source_subdomains = [x.split(datasets.DATASET_SEPARATOR)[1] for x in args.source_dataset]
    target_subdomains = [x.split(datasets.DATASET_SEPARATOR)[1] for x in args.target_dataset]
    assert len(set(source_datasets)) == 1, "Can only load from same datasets!"
    dataset_name = args.source_dataset[0].split(datasets.DATASET_SEPARATOR)[0]
    c['current'] = c['datasets'][dataset_name]

    # Make logdir
    args.exp_dir = osp.join(c['paths']['log_dir'],
                            dataset_name,
                            c['architecture'],
                            remove_extension(args.sub_dir.split('/')[-1]),
                            )
    args.results_file = osp.join(args.exp_dir, 'results_{}.csv')
    args.task_dir = osp.join(args.exp_dir,
                             "{}-{}_{}_{}".format(','.join(source_subdomains),
                                                  ','.join(target_subdomains),
                                                  str(datetime.datetime.now()).replace(' ', '_'),
                                                  socket.gethostname(),
                                                  ),
                             )
    args.feature_file = osp.join(args.task_dir, 'features', 'features_{}.pth')
    os.makedirs(osp.dirname(args.feature_file), exist_ok=True, mode=0o755)
    args.snapshot_path = osp.join(args.task_dir, 'snapshots', 'snapshot.pth')
    os.makedirs(osp.dirname(args.snapshot_path), exist_ok=True, mode=0o755)

    # Fix path when loading from snapshot
    if args.snapshot is not None and not args.snapshot.endswith('.pth'):
        args.snapshot = osp.join(args.snapshot, 'snapshots', 'snapshot.pth')

    config.merge_into_global_from_dict(args.__dict__)
    u.save_setup()
    return args
