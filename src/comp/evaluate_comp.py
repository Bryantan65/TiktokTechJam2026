"""Score a trained checkpoint under the TechJam Track 2 rules.

Differs from CWM's own evaluation in four ways, all required by the competition
spec and all switchable from the config below:
  label        is_click          (CWM uses long_view2)
  ndcg_k       10                (CWM uses 1/3/5)
  recall_k     50                (CWM computes no recall at all)
  split        50/50 by time     (CWM cuts 04-22..04-28 / 04-29..05-08)

Unknowns still pending organiser confirmation are config switches rather than
hardcoded assumptions, so the 28 Aug answers become a config edit.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# make src/ importable regardless of the directory this is launched from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_wrapper import Wrap_Dataset
from utils.summary_dat import make_feature, cal_field_dims
from comp.metrics import ndcg_at_k, recall_at_k, aggregate

SUBSET = '../rec_datasets/processed/KuaiRand_subset.csv'

FEATURE_COLS = ['user_id', 'follow_user_num_range', 'register_days_range',
                'fans_user_num_range', 'friend_user_num_range', 'user_active_degree',
                'video_id', 'author_id', 'music_id', 'tag_pop', 'video_type',
                'upload_type', 'tab']

DEFAULTS = {
    'label': 'is_click',
    'ndcg_k': 10,
    'recall_k': 50,
    'split_point': 0.5,               # fraction of ROWS, sorted by time_ms
    'candidate_set': 'impressions',   # | full_catalog | sampled_negatives
    'zero_positive': 'skip',          # | zero
}


def load_splits(cfg, label):
    """train = 04-08..04-21; the 04-22..05-08 window is cut 50/50 by time."""
    cols = sorted(set(FEATURE_COLS + ['date', 'time_ms', label]))
    dat = pd.read_csv(SUBSET, usecols=cols)

    train = dat[(dat['date'] >= 20220408) & (dat['date'] <= 20220421)]
    later = dat[(dat['date'] >= 20220422) & (dat['date'] <= 20220508)]
    later = later.sort_values('time_ms', kind='mergesort').reset_index(drop=True)

    cut = int(len(later) * cfg['split_point'])
    return dat, train, later.iloc[:cut], later.iloc[cut:]


def build_model(model_name, field_dims):
    from model.fm import My_FactorizationMachineModel
    from model.dfm import My_DeepFactorizationMachineModel
    from model.dcn import My_DeepCrossNetworkModel
    from model.xdfm import My_ExtremeDeepFactorizationMachineModel
    from model.afi import My_AutomaticFeatureInteractionModel

    if model_name == 'FM':
        return My_FactorizationMachineModel(field_dims=field_dims, embed_dim=10)
    if model_name == 'DFM':
        return My_DeepFactorizationMachineModel(field_dims=field_dims, embed_dim=10,
                                                mlp_dims=[64, 64, 64], dropout=0.2)
    if model_name == 'DCN':
        return My_DeepCrossNetworkModel(field_dims=field_dims, embed_dim=10, num_layers=3,
                                        mlp_dims=[64, 64, 64], dropout=0.2)
    if model_name == 'xDFM':
        return My_ExtremeDeepFactorizationMachineModel(field_dims=field_dims, embed_dim=10,
                                                       mlp_dims=[64, 64, 64], dropout=0.2,
                                                       cross_layer_sizes=[64, 64, 64],
                                                       split_half=True)
    if model_name == 'AFI':
        return My_AutomaticFeatureInteractionModel(field_dims=field_dims, embed_dim=10,
                                                   num_heads=8, num_layers=1,
                                                   atten_embed_dim=64, mlp_dims=[64],
                                                   dropouts=[0.2, 0.2])
    raise ValueError('unknown model_name: ' + str(model_name))


def predict(model, df, label, batch_size=4096, use_cuda=True):
    ds = Wrap_Dataset(make_feature(df, 'KuaiRand'), df[label].tolist(), use_cuda=use_cuda)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        model.eval()
        for batch in ld:
            out.extend(model(batch[0]).view(batch[0].size(0)).cpu().tolist())
    return np.asarray(out)


def score(df, preds, cfg):
    """Rank each user's candidates by prediction, then average the metrics."""
    if cfg['candidate_set'] != 'impressions':
        raise NotImplementedError(
            'candidate_set=' + str(cfg['candidate_set']) + ' not implemented. Only each '
            'user logged impressions are supported; full_catalog needs negative sampling '
            'and a catalogue-wide scoring pass. Pending organiser confirmation.')

    label = cfg['label']
    work = pd.DataFrame({'user_id': df['user_id'].values,
                         'label': df[label].values,
                         'pred': preds})
    work = work.sort_values(['user_id', 'pred'], ascending=[True, False], kind='mergesort')

    ndcgs, recalls, sizes = [], [], []
    for _, grp in work.groupby('user_id', sort=False):
        lab = grp['label'].values
        ndcgs.append(ndcg_at_k(lab, cfg['ndcg_k']))
        recalls.append(recall_at_k(lab, cfg['recall_k']))
        sizes.append(len(lab))

    zp = cfg['zero_positive']
    return {
        'ndcg@' + str(cfg['ndcg_k']): aggregate(ndcgs, zp),
        'recall@' + str(cfg['recall_k']): aggregate(recalls, zp),
        'n_users': len(sizes),
        'n_rows': int(len(work)),
        'users_with_no_positive': int(sum(1 for v in recalls if v is None)),
        'median_candidates_per_user': float(np.median(sizes)),
    }


def main():
    ap = argparse.ArgumentParser(description='Score a checkpoint under competition rules')
    ap.add_argument('--fout', required=True, help='checkpoint prefix, as passed to main.py')
    ap.add_argument('--model_name', default='FM', choices=['FM', 'DFM', 'DCN', 'xDFM', 'AFI'])
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--allow_test', action='store_true',
                    help='required to score the held-out test half')
    ap.add_argument('--checkpoint_suffix', default='_model.pt')
    ap.add_argument('--out', default=None, help='write JSON here')
    for key, val in DEFAULTS.items():
        ap.add_argument('--' + key, default=val, type=type(val))
    args = ap.parse_args()

    cfg = {k: getattr(args, k) for k in DEFAULTS}

    if args.split == 'test' and not args.allow_test:
        print(json.dumps({'status': 'error', 'error':
              'refusing to score the test half without --allow_test; develop on val'}))
        sys.exit(2)

    t0 = time.time()
    try:
        all_dat, _train, val, test = load_splits(cfg, cfg['label'])
        df = val if args.split == 'val' else test

        model = build_model(args.model_name, cal_field_dims(all_dat, 'KuaiRand'))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            model = model.cuda()
        model.load_state_dict(torch.load(args.fout + args.checkpoint_suffix,
                                         map_location='cuda' if use_cuda else 'cpu'))

        preds = predict(model, df, cfg['label'], use_cuda=use_cuda)
        result = score(df, preds, cfg)
        result.update({'status': 'ok', 'error': None})
    except Exception as exc:
        result = {'status': 'error', 'error': type(exc).__name__ + ': ' + str(exc)}

    result['gpu_seconds'] = round(time.time() - t0, 2)
    result['split'] = args.split
    result['config'] = cfg

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, 'w') as fh:
            fh.write(text)


if __name__ == '__main__':
    main()
