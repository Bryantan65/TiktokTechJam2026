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

Both `load` and `encode` are cached on disk. Neither changes what it returns;
they only avoid recomputing it. Measured cost of a cold call, this machine:

    Pure   1,436,609 rows   load  3.9s + encode  6.4s =  10.3s
    1k    11,713,045 rows   load 48.7s + encode 54.1s = 102.8s

That is paid by every seed of every experiment, and every seed runs in its own
subprocess, so a three-seed experiment pays it three times over. Against the
archive's per-experiment compute it is 4.1% of all Pure time and roughly a
third of all 1k time - on 1k no experiment finishes in under 60s, and the
median one spends a third of itself here before a model exists.

Correctness comes first, because a stale or mismatched cache would not crash -
it would train on the wrong thing and report a plausible number.

  - `load` is keyed on the directory and invalidated by source mtime. Its only
    input is `data_dir`, so this is exact.
  - `encode` is keyed on a hash of the splits it was actually handed, not on
    the directory. Solutions are free to filter or augment splits before
    encoding, and a directory-keyed cache would silently hand such a solution
    the canonical encoding instead. Hashing costs a fraction of encoding.

Set HARNESS_CACHE=0 to bypass both, HARNESS_CACHE_DIR to relocate them.
"""
import csv
import hashlib
import importlib.util as _il
import os
import pickle
import tempfile

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
_kit_encode = _kit.encode
_bucket_edges = _kit._bucket_edges

# Suffix used in every filename of a variant. Pure is the required benchmark;
# 1k and 27k are the bonus ones. 27k ships its standard logs in two parts.
_VARIANTS = ('pure', '1k', '27k')


# Bump when the cached payload's meaning changes; old files then simply miss.
_SCHEMA = 1
_CACHE_ON = os.environ.get('HARNESS_CACHE', '1') != '0'
_CACHE_DIR = os.environ.get('HARNESS_CACHE_DIR') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache')


# Fraction of the TRAINING split to keep, 1.0 for all of it. Never touches valid
# or test: those are what the run is scored on and must stay whole.
#
# For 27k only. A full-data experiment there costs ~115 min - 26 min to load, 12
# to encode, 77 to train - against a 6 h ceiling, so a run fits three
# experiments and every result we have came from experiment 23 or later. The
# training split is 139M rows against a model with ~2M parameters, roughly 70
# rows per parameter, so most of it is redundant capacity rather than signal.
#
# Sampling is deterministic in the row index, not random per process, so every
# seed and every solution in a run sees exactly the same subset - otherwise
# experiments stop being comparable to each other, which is the one thing the
# ledger depends on.
#
# It is a disclosed resource decision, not a free win: the baseline is measured
# on full data, so an agent trained on a fraction starts behind it. Measure the
# cost before trusting a run that uses this.
TRAIN_FRACTION = float(os.environ.get('HARNESS_TRAIN_FRACTION', 1.0))


def _subsample_train(splits):
    """Keep every Nth training row. Deterministic, and valid/test untouched."""
    f = TRAIN_FRACTION
    if not (0.0 < f < 1.0):
        return splits
    step = max(2, int(round(1.0 / f)))
    tr = splits.get('train') or []
    splits['train'] = tr[::step]
    print('data.py: HARNESS_TRAIN_FRACTION=%.4f - training split %d -> %d rows '
          '(every %dth); valid and test untouched'
          % (f, len(tr), len(splits['train']), step))
    return splits


def _cache_path(name):
    return os.path.join(_CACHE_DIR, '%s_v%d.pkl' % (name, _SCHEMA))


def _cache_get(path, newer_than=()):
    """Return the cached object, or None if absent, stale or unreadable.

    `newer_than` are source files the cache must post-date. A corrupt or
    half-written file is treated as a miss rather than an error: the cache is
    an optimisation and must never be able to fail a run.
    """
    if not _CACHE_ON or not os.path.isfile(path):
        return None
    try:
        stamp = os.path.getmtime(path)
        for src_file in newer_than:
            if os.path.getmtime(src_file) > stamp:
                return None
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _cache_put(path, obj):
    """Write atomically, because the seeds of one experiment race here.

    Every seed is a separate subprocess and they start together, so on a cold
    cache all of them compute the same value and all try to store it. Writing
    to a temporary file and renaming means a reader never observes a partial
    file; the writers are interchangeable, so last-one-wins is correct.
    """
    if not _CACHE_ON:
        return
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as fh:
                pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
    except Exception:
        pass        # an unwritable cache is a slow run, not a broken one


def _sources(data_dir):
    """Files whose modification invalidates a cached load of `data_dir`."""
    v = variant(data_dir)
    out = [os.path.join(data_dir, 'video_features_basic_%s.csv' % v)]
    try:
        out += _log_files(data_dir, v)
    except FileNotFoundError:
        pass
    return [p for p in out if os.path.isfile(p)]


def _dir_tag(data_dir):
    h = hashlib.blake2b(os.path.abspath(data_dir).encode('utf-8'),
                        digest_size=6).hexdigest()
    return '%s_%s' % (variant(data_dir), h)


def encode(splits):
    """The kit's encode, memoised on the content of `splits`.

    Keyed on the splits themselves rather than on the data directory: a
    solution may hand this filtered or augmented splits, and those must not
    collide with the canonical encoding. Hashing the pickled splits is the
    price of that guarantee, and it is far below the cost of encoding.
    """
    if not _CACHE_ON:
        return _kit_encode(splits)
    try:
        blob = pickle.dumps(splits, protocol=pickle.HIGHEST_PROTOCOL)
        tag = hashlib.blake2b(blob, digest_size=16).hexdigest()
    except Exception:
        return _kit_encode(splits)      # unpicklable splits: just encode
    path = _cache_path('encode_%s' % tag)
    hit = _cache_get(path)
    if hit is not None:
        return hit
    out = _kit_encode(splits)
    _cache_put(path, out)
    return out


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


def _load_uncached(data_dir, only=None):
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

    def _num(x, default=0.0):
        """Coerce a CSV cell, tolerating a malformed row.

        csv.DictReader hands back None for any field a short row does not
        reach, so a single ragged line makes float() raise and takes the whole
        load down with it. KuaiRand-1k has exactly one such row in
        log_standard_4_08_to_4_21_1k.csv - 19 header columns, one line with a
        different count - and it was enough to break every 1k experiment before
        a model was ever built. Pure has none, which is why this only ever
        showed up on the bonus variants.

        Defaulted rather than skipped, deliberately: a solution's prediction
        array is indexed by row position, so dropping a row here would silently
        misalign every downstream score.
        """
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    # Rows outside `only` are skipped as they are read, never materialised.
    # Every row becomes a Python tuple of seven objects - roughly 340 bytes
    # against ~150 on disk - so on 27k the full 322M rows need ~110 GB and a
    # container capped at 116 GB dies mid-load with no traceback. train+valid
    # is 208M rows, which fits. Deliverable 4 asks for validation-best scores,
    # so the test split is not needed to measure a baseline.
    keep = None
    if only:
        keep = [SPLITS[n] for n in only if n in SPLITS]

    rows = []
    bad = 0
    for path in _log_files(data_dir, v):
        with open(path, encoding='utf-8', newline='') as fh:
            for r in csv.DictReader(fh):
                dur = _num(r.get('duration_ms'))
                date = r.get('date')
                if dur == 0.0 and not r.get('duration_ms'):
                    bad += 1
                d = int(_num(date))
                if keep is not None and not any(lo <= d <= hi for lo, hi in keep):
                    continue
                rows.append((d, r.get('user_id') or '',
                             r.get('video_id') or '',
                             vid2author.get(r.get('video_id'), 'UNK'),
                             r.get('tab') or '', dur,
                             1 if (r.get(LABEL) or '0') != '0' else 0))
    if bad:
        print('data.py: %d malformed row(s) in %s logs - numeric fields '
              'defaulted, row order preserved' % (bad, v))

    return {name: [x for x in rows if lo <= x[0] <= hi]
            for name, (lo, hi) in SPLITS.items()}


def load(data_dir, only=None):
    """Splits dict for `data_dir`, reusing a cached parse when one is current.

    `only` is an optional iterable of split names. When given, rows outside
    those splits are skipped as the CSVs are read rather than built and
    discarded - the difference between 110 GB and 71 GB on 27k. Omit it and
    behaviour is exactly as before.

    Cache-invalidating on source mtime rather than on a content hash: reading
    every CSV to hash it would cost what the parse costs and save nothing.
    """
    tag = _dir_tag(data_dir)
    if only:
        tag += '_' + '-'.join(sorted(only))
    path = _cache_path('load_%s' % tag)
    hit = _cache_get(path, newer_than=_sources(data_dir))
    if hit is not None:
        return _subsample_train(hit)
    out = _load_uncached(data_dir, only=only)
    _cache_put(path, out)
    return _subsample_train(out)
