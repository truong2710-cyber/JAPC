#!/usr/bin/env python3
"""
Compare validation JSON results and compute p-value between (base_mean of two sets) and (ours_mean of two sets).

Usage:
  python compare_validation_pvalue.py base1.json base2.json ours1.json ours2.json [--test ttest|wilcoxon]

The script expects each JSON to be a list of entries with `support_idx` and
`mar_val_batches_classDice` -> `values` containing nested numeric lists.
For each `support_idx` we compute the mean across all numbers in `values`.
Then for each support_idx present in all four files we compute
  base = (mean_base1 + mean_base2) / 2
  ours = (mean_ours1 + mean_ours2) / 2
and perform a paired statistical test between `ours` and `base`.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from typing import Dict, Iterable, List


def flatten_numbers(x: Iterable) -> List[float]:
    out: List[float] = []
    if isinstance(x, (int, float)):
        return [float(x)]
    try:
        for v in x:
            if isinstance(v, (int, float)):
                out.append(float(v))
            else:
                out.extend(flatten_numbers(v))
    except TypeError:
        pass
    return out


def extract_support_means(path: str) -> Dict[int, float]:
    with open(path, 'r') as f:
        data = json.load(f)
    mapping: Dict[int, float] = {}
    for entry in data:
        if 'support_idx' not in entry:
            continue
        idx = int(entry['support_idx'])
        # defensive access
        values = entry.get('mar_val_batches_classDice', {}).get('values', [])
        nums = flatten_numbers(values)
        if not nums:
            continue
        mapping[idx] = float(sum(nums) / len(nums))
    return mapping


def paired_permutation_test(a: List[float], b: List[float], n_permutations: int = 10000, seed: int | None = 0) -> float:
    # two-sided permutation test on paired differences
    if seed is not None:
        random.seed(seed)
    diffs = [x - y for x, y in zip(a, b)]
    obs = abs(sum(diffs) / len(diffs))
    count = 0
    for _ in range(n_permutations):
        signs = [random.choice((1, -1)) for _ in diffs]
        perm_mean = abs(sum(d * s for d, s in zip(diffs, signs)) / len(diffs))
        if perm_mean >= obs:
            count += 1
    return (count + 1) / (n_permutations + 1)


def main():
    p = argparse.ArgumentParser(description='Compare validation JSONs and compute paired p-value')
    p.add_argument('base1')
    p.add_argument('base2')
    p.add_argument('ours1')
    p.add_argument('ours2')
    p.add_argument('--test', choices=('ttest', 'wilcoxon'), default='ttest')
    p.add_argument('--permutations', type=int, default=10000,
                   help='permutations used if scipy not available (default: 10000)')
    args = p.parse_args()

    m_base1 = extract_support_means(args.base1)
    m_base2 = extract_support_means(args.base2)
    m_ours1 = extract_support_means(args.ours1)
    m_ours2 = extract_support_means(args.ours2)

    common = sorted(set(m_base1) & set(m_base2) & set(m_ours1) & set(m_ours2))
    if not common:
        raise SystemExit('No common support_idx found across the four files')

    base_list: List[float] = []
    ours_list: List[float] = []
    for idx in common:
        base_list.append((m_base1[idx] + 2.0 * m_base2[idx]) / 3.0)
        ours_list.append((m_ours1[idx] + 2.0 * m_ours2[idx]) / 3.0)
    # breakpoint()  
    # attempt to use scipy for tests
    try:
        import numpy as _np  # noqa: N813
        from scipy import stats  # type: ignore
        if args.test == 'ttest':
            res = stats.ttest_rel(ours_list, base_list, alternative='greater')
            print(f"paired t-test statistic={res.statistic:.6f}, pvalue={res.pvalue:.6e}")
        else:
            # wilcoxon requires non-equal length handling; SciPy's wilcoxon supports paired
            res = stats.wilcoxon(ours_list, base_list)
            print(f"wilcoxon statistic={res.statistic:.6f}, pvalue={res.pvalue:.6e}")
        # Print mean and std of gain
        gain = _np.array(ours_list) - _np.array(base_list)
        print(f"mean(ours - base) = {gain.mean():.6f}, std(ours - base) = {gain.std(ddof=1):.6f}")
    except Exception:
        # fallback: permutation test for paired differences
        pval = paired_permutation_test(ours_list, base_list, n_permutations=args.permutations)
        # also compute simple mean difference
        mean_diff = sum(x - y for x, y in zip(ours_list, base_list)) / len(base_list)
        print(f"scipy not available or test failed — using permutation fallback")
        print(f"mean(ours - base) = {mean_diff:.6f}, pvalue(permutation) = {pval:.6e}")


if __name__ == '__main__':
    main()
