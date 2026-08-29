"""Variant-aware loader: KuaiRand-Pure, -1k, -27k, without touching the kit.

`harness/` sits ahead of the kit on the subprocess PYTHONPATH, so a solution's
`from data import load` resolves here. The starter kit stays byte-identical,
which is the point: `kuairand-starter-kit/` is read-only, and a judge diffing it
against the official release should find nothing.

Pure is not reimplemented. `load()` delegates to the kit's own function for the
required benchmark, so Pure behaviour is unchanged by construction rather than
by testing. Only the bonus variants take a different path, and even that path
differs from the kit's only in which filenames it opens.

Everything else - LABEL, FIELDS, SPLITS, encode, the duration bucketing - is
re-exported from the kit unchanged, so solutions see exactly one implementation.
"""
import csv
import importlib.util as _il
import os

_KIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'kuairand-starter-kit')

# Load the kit's data.py by path under a different module name. Importing it as
# `data` would resolve back to this file, since harness/ is earlier on the path.
_spec = _il.spec_from_file_location('_kuairand_kit_data', os.path.join(_KIT, 'data.py'))
_kit = _il.module_from_spec(_spec)
_spec.loader.exec_module(_kit)

# Re-exported unchanged. Solutions and the harness see the kit's definitions.
LABEL = _kit.LABEL
SPLITS = _kit.SPLITS
FIELDS = _kit.FIELDS
encode = _kit.encode
_bucket_edges = _kit._bucket_edges

# Suffix used in every filename of a variant. Pure is the required benchmark;
# 1k and 27k are the bonus ones. 27k ships its standard logs in two parts.
_VARIANTS = ('pure', '1k', '27k')


def variant(data_dir):
    """Which KuaiRand variant `data_dir` holds, by the files actually present.

    Detection is by the video-features file, which every variant ships exactly
    one of. Falls back to 'pure' so an unrecognised directory fails loudly in
    the kit's own loader rather than silently loading something unintended.
    """
    for v in ('27k', '1k'):
        if os.path.isfile(os.path.join(data_dir, 'video_features_basic_%s.csv' % v)):
            return v
    return 'pure'


def _log_files(data_dir, v):
    """Standard logs for a variant, in date order. 27k splits each into parts."""
    out = []
    for window in ('4_08_to_4_21', '4_22_to_5_08'):
        single = os.path.join(data_dir, 'log_standard_%s_%s.csv' % (window, v))
        if os.path.isfile(single):
            out.append(single)
            continue
        part, i = [], 1
        while True:
            p = os.path.join(data_dir, 'log_standard_%s_%s_part%d.csv' % (window, v, i))
            if not os.path.isfile(p):
                break
            part.append(p)
            i += 1
        if not part:
            raise FileNotFoundError('no standard log for window %s in %s' % (window, data_dir))
        out.extend(part)
    return out


def load(data_dir):
    """Splits dict for whichever variant `data_dir` holds.

    Pure goes through the kit's own loader untouched. The bonus variants use the
    same split dates, the same row tuple and the same author lookup - only the
    filenames differ.
    """
    v = variant(data_dir)
    if v == 'pure':
        return _kit.load(data_dir)

    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_%s.csv' % v),
              encoding='utf-8', newline='') as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for path in _log_files(data_dir, v):
        with open(path, encoding='utf-8', newline='') as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0))

    return {name: [x for x in rows if lo <= x[0] <= hi]
            for name, (lo, hi) in SPLITS.items()}
