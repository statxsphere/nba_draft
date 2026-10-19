# which feature upgrades earn their place: fixed settings, one change at a time, scored on the same
# outer classes as the backtest. this is the one step that looks at the test classes, so it only
# switches whole groups on or off, nothing is tuned here
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from train import (load_data, fit_blend, predict_parts, TARGETS, FIRST_TEST, V3_PARAMS, FINAL)

FIXED = {'gbm': dict(V3_PARAMS), 'alpha': 30, 'w': 0.5}
KEYS = ['peak', 'imm', 'bpm']   # the feature choices were made on these three, before the rapm targets existed

V3 = dict(seasons='last', schedule=False, shrink=False, age=False, position=False, flags=False, shots=False)
FULL = dict(seasons='last', schedule=True, shrink=True, age=True, position=True, flags=True, shots=True)
CHOSEN = dict(V3, seasons='new')   # without shot types, which are tested on top of it

RUNS = {
    'v3 inputs': V3,
    '+ schedule': dict(V3, schedule=True),
    '+ shrinkage': dict(V3, shrink=True),
    '+ age adjusted': dict(V3, age=True),
    '+ position relative': dict(V3, position=True),
    '+ combine flags': dict(V3, flags=True),
    '+ seasons 5/3/2 newest': dict(V3, seasons='new'),
    '+ seasons 2/3/5 oldest': dict(V3, seasons='old'),
    '+ shot types': dict(V3, shots=True),
    'all': FULL,
    'all - schedule': dict(FULL, schedule=False),
    'all - shrinkage': dict(FULL, shrink=False),
    'all - age adjusted': dict(FULL, age=False),
    'all - position relative': dict(FULL, position=False),
    'all - combine flags': dict(FULL, flags=False),
    'all - shot types': dict(FULL, shots=False),
    'all, seasons 5/3/2 newest': dict(FULL, seasons='new'),
    'all, seasons 2/3/5 oldest': dict(FULL, seasons='old'),
    'chosen': CHOSEN,
    'chosen + shot types': dict(CHOSEN, shots=True),
}


def one(df, feats, key, year):
    target = TARGETS[key]
    done = df[df[target].notna()]
    train = done[done['draft_year'] < year]
    test = done[done['draft_year'] == year]
    g, r = predict_parts(fit_blend(train, target, feats, FIXED), test)
    y = test[target].values
    return {'target': key, 'year': year, 'gbm': spearmanr(g, y)[0], 'ridge': spearmanr(r, y)[0],
            'blend': spearmanr(0.5 * g + 0.5 * r, y)[0]}


def main():
    rows = []
    for name, opts in RUNS.items():
        df, feats = load_data(opts)
        jobs = []
        for key in KEYS:
            last = int(df[df[TARGETS[key]].notna()]['draft_year'].max())
            jobs += [(key, y) for y in range(FIRST_TEST, last + 1)]
        out = Parallel(n_jobs=24)(delayed(one)(df, feats, k, y) for k, y in jobs)
        for o in out:
            o['run'] = name
        rows += out
        t = pd.DataFrame(out).groupby('target')[['gbm', 'ridge', 'blend']].mean()
        print(f"{name:<28}" + '  '.join(f"{k} {t.loc[k, 'gbm']:.3f}/{t.loc[k, 'ridge']:.3f}/{t.loc[k, 'blend']:.3f}"
                                       for k in KEYS), flush=True)
    res = pd.DataFrame(rows)

    # classes won against the base of each comparison, on the blend
    out = []
    for name in RUNS:
        base = 'v3 inputs' if not name.startswith('all') or name == 'all' else 'all'
        base = 'chosen' if name.startswith('chosen') else base
        a = res[res['run'] == name].set_index(['target', 'year'])
        b = res[res['run'] == base].set_index(['target', 'year'])
        r = {'run': name, 'compared_with': base}
        for key in KEYS:
            for m in ['gbm', 'ridge', 'blend']:
                r[f'{key}_{m}'] = a.loc[key, m].mean()
        r['mean_blend'] = np.mean([r[f'{k}_blend'] for k in KEYS])
        r['mean_change'] = (a['blend'] - b['blend']).mean()
        r['classes_better'] = int((a['blend'] > b['blend']).sum())
        r['classes'] = len(a)
        # shot types only exist from the 2014 class on, so they are also judged on those classes alone
        late = a.index.get_level_values('year') >= 2014
        r['change_2014_on'] = (a['blend'] - b['blend'])[late].mean()
        r['better_2014_on'] = int((a['blend'] > b['blend'])[late].sum())
        r['classes_2014_on'] = int(late.sum())
        out.append(r)
    out = pd.DataFrame(out)
    print(out[['run', 'compared_with', 'mean_blend', 'mean_change', 'classes_better', 'classes', 'change_2014_on',
               'better_2014_on', 'classes_2014_on']].round(3).to_string(index=False))
    out.round(4).to_csv(os.path.join(FINAL, 'ablation.csv'), index=False)


if __name__ == '__main__':
    main()
