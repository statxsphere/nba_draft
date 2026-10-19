# my own nba plus-minus: every 5 man lineup's points for and against per 100 possessions, from
# stats.nba.com, then a ridge regression on who was on the floor. 3 season windows, with low minute
# players pulled toward replacement level. run it after build_dataset.py, it adds peak_rapm and
# imm_rapm to draft_players.csv
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, 'data', 'raw')
FINAL = os.path.join(ROOT, 'data', 'final')

FIRST_SEASON = 2008     # 2007-08, the first season stats.nba.com has lineups for
LAST_SEASON = 2026
PRIOR = -2.0            # replacement level, same as for bpm
LAMBDA = 1500           # ridge strength in possessions, picked by cross validation on lineups below

RAPTOR = 'https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/modern_RAPTOR_by_player.csv'


def season_name(year):
    return f'{year - 1}-{str(year)[-2:]}'


def get_lineups(year):
    from nba_api.stats.endpoints import leaguedashlineups
    from nba_api.stats.static import teams
    path = os.path.join(RAW, 'lineups', f'lineups_{year}.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    frames = []
    for t in teams.get_teams():
        # one team at a time, a league wide call stops at 2000 lineups
        part = os.path.join(RAW, 'lineups', f'lineups_{year}_{t["id"]}.csv')
        if not os.path.exists(part):
            for attempt in range(3):
                try:
                    time.sleep(1.5)
                    df = leaguedashlineups.LeagueDashLineups(
                        season=season_name(year), team_id_nullable=str(t['id']), group_quantity=5,
                        measure_type_detailed_defense='Advanced', per_mode_detailed='Totals',
                        timeout=90).get_data_frames()[0]
                    df.to_csv(part, index=False)
                    break
                except Exception as e:
                    print(f'  {year} {t["abbreviation"]} failed ({e}), waiting..')
                    time.sleep(15)
            else:
                raise RuntimeError(f'could not get lineups for {year} {t["abbreviation"]}')
        frames.append(pd.read_csv(part))
    df = pd.concat(frames)
    df = df[['GROUP_ID', 'TEAM_ID', 'MIN', 'POSS', 'OFF_RATING', 'DEF_RATING']]
    df['season'] = year
    df.to_csv(path, index=False)
    for t in teams.get_teams():
        os.remove(os.path.join(RAW, 'lineups', f'lineups_{year}_{t["id"]}.csv'))
    print(f'  lineups {year}: {len(df)}')
    return df


def get_all_lineups():
    os.makedirs(os.path.join(RAW, 'lineups'), exist_ok=True)
    return pd.concat([get_lineups(y) for y in range(FIRST_SEASON, LAST_SEASON + 1)], ignore_index=True)


def design(lineups):
    # one row per lineup, a 1 for each of the five players on the floor
    ids = lineups['GROUP_ID'].str.strip('-').str.split('-')
    players = sorted({int(p) for g in ids for p in g})
    idx = {p: i for i, p in enumerate(players)}
    rows = np.repeat(np.arange(len(ids)), 5)
    cols = [idx[int(p)] for g in ids for p in g]
    X = sparse.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(ids), len(players)))
    return X, np.array(players)


def centred(lineups):
    # points scored and allowed per 100, against the league average that season
    d = lineups.copy()
    for c in ['OFF_RATING', 'DEF_RATING']:
        avg = (d[c] * d['POSS']).groupby(d['season']).sum() / d['POSS'].groupby(d['season']).sum()
        d[c] = d[c] - d['season'].map(avg)
    return d


def fit_window(lineups, lam=LAMBDA):
    # ridge on the gap from the prior: a player with no minutes stays at replacement level
    from sklearn.linear_model import Ridge
    d = centred(lineups[lineups['POSS'] > 0])
    X, players = design(d)
    w = d['POSS'].values.astype(float)
    out = pd.DataFrame({'person_id': players})
    for side, col, prior in [('o', 'OFF_RATING', PRIOR / 2), ('d', 'DEF_RATING', -PRIOR / 2)]:
        y = d[col].values - 5 * prior
        r = Ridge(alpha=lam, fit_intercept=False).fit(X, y, sample_weight=w)
        out[side] = prior + r.coef_
    out['rapm'] = out['o'] - out['d']
    out['poss'] = X.T @ w
    return out


def pick_lambda(lineups):
    # 5 fold cross validation on lineups, which never looks at the draft at all
    d = centred(lineups[lineups['POSS'] > 0]).reset_index(drop=True)
    fold = np.random.RandomState(7).randint(0, 5, len(d))
    best = None
    for lam in [250, 500, 1000, 1500, 2500, 4000, 6000]:
        err = 0
        for k in range(5):
            fit = fit_window(d[fold != k], lam)
            t = d[fold == k]
            X, players = design(t)
            v = fit.set_index('person_id').reindex(players)
            o = X @ v['o'].fillna(PRIOR / 2).values
            dd = X @ v['d'].fillna(-PRIOR / 2).values
            err += np.sum(t['POSS'] * ((t['OFF_RATING'] - o) ** 2 + (t['DEF_RATING'] - dd) ** 2))
        print(f'  lambda {lam}: {err / d["POSS"].sum():.1f}')
        if best is None or err < best[1]:
            best = (lam, err)
    return best[0]


def rapm_windows(lineups, length, lam=LAMBDA):
    out = []
    for start in range(FIRST_SEASON, LAST_SEASON - length + 2):
        w = fit_window(lineups[lineups['season'].between(start, start + length - 1)], lam)
        w['start'] = start
        w['length'] = length
        out.append(w)
    return pd.concat(out, ignore_index=True)


def check_raptor(windows, df):
    # rough check against fivethirtyeight's modern raptor (cc by 4.0, 2013-14 to 2021-22), same 3 season windows
    path = os.path.join(RAW, 'raptor_by_player.csv')
    if not os.path.exists(path):
        import requests
        r = requests.get(RAPTOR, timeout=120)
        r.raise_for_status()
        open(path, 'wb').write(r.content)
    rap = pd.read_csv(path)
    rap = rap[rap['season_type'] == 'RS'] if 'season_type' in rap else rap
    ids = df[['person_id', 'bbref_id']].dropna()
    rows = []
    w3 = windows[(windows['length'] == 3) & (windows['poss'] >= 3000)].merge(ids, on='person_id')
    for _, g in w3.groupby('start'):
        start = g['start'].iloc[0]
        r = rap[rap['season'].between(start, start + 2)]
        r = r.groupby('player_id').apply(lambda x: pd.Series({
            'raptor': np.average(x['raptor_total'], weights=x['poss']),
            'raptor_o': np.average(x['raptor_offense'], weights=x['poss']),
            'raptor_d': np.average(x['raptor_defense'], weights=x['poss'])}), include_groups=False)
        rows.append(g.merge(r, left_on='bbref_id', right_index=True))
    m = pd.concat(rows)
    res = {'players': int(m['person_id'].nunique()), 'player_windows': len(m),
           'r_total': m['rapm'].corr(m['raptor']), 'r_offence': m['o'].corr(m['raptor_o']),
           'r_defence': (-m['d']).corr(m['raptor_d'])}
    print('  against raptor: ' + ', '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in res.items()))
    return res


def add_targets(df, windows):
    # peak: best 3 season window inside seasons 2 to 6, like peak bpm. immediate: seasons 1 and 2.
    # a window he didn't play in counts as replacement level, and nothing counts below it
    w3 = windows[windows['length'] == 3].set_index(['person_id', 'start'])['rapm']
    w2 = windows[windows['length'] == 2].set_index(['person_id', 'start'])['rapm']
    peak, imm = [], []
    for _, p in df.iterrows():
        y = p['draft_year']
        starts = [s for s in range(y + 2, y + 5) if FIRST_SEASON <= s and s + 2 <= LAST_SEASON]
        if y + 6 <= LAST_SEASON and len(starts) >= 2:
            vals = [w3.get((p['person_id'], s), PRIOR) for s in starts]
            peak.append(max(max(vals), PRIOR))
        else:
            peak.append(np.nan)
        s = y + 1
        if s >= FIRST_SEASON and s + 1 <= LAST_SEASON:
            imm.append(max(w2.get((p['person_id'], s), PRIOR), PRIOR))
        else:
            imm.append(np.nan)
    df['peak_rapm'] = np.round(peak, 3)
    df['imm_rapm'] = np.round(imm, 3)
    return df


def main():
    import json
    lineups = get_all_lineups()
    print(f'{len(lineups)} lineup seasons, {lineups["POSS"].sum() / 2:.0f} possessions each way')
    lam = pick_lambda(lineups[lineups['season'].between(2014, 2016)])
    print(f'  lambda {lam}')
    windows = pd.concat([rapm_windows(lineups, 3, lam), rapm_windows(lineups, 2, lam)])
    df = pd.read_csv(os.path.join(FINAL, 'draft_players.csv'))
    check = check_raptor(windows, df)
    check['lambda'] = lam
    # only drafted players are kept, the rest of the league is just there to fit against
    windows = windows[windows['person_id'].isin(df['person_id'])]
    windows.round(3).to_csv(os.path.join(FINAL, 'nba_rapm.csv'), index=False)
    df = add_targets(df, windows)
    df.to_csv(os.path.join(FINAL, 'draft_players.csv'), index=False)
    print(f"peak rapm for {df['peak_rapm'].notna().sum()} players, immediate rapm for {df['imm_rapm'].notna().sum()}")
    with open(os.path.join(FINAL, 'rapm_check.json'), 'w') as f:
        json.dump(check, f, indent=1)
    print('Done!')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fetch':
        get_all_lineups()
    else:
        main()
