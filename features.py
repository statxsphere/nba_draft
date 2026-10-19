import os
import json

import numpy as np
import pandas as pd

from build_dataset import rates, pos_group, RATE_STATS, EXPOSURE, COUNTS, TEAM_COUNTS

ROOT = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(ROOT, 'data', 'final')

BASE = ['age', 'height', 'weight', 'wingspan', 'wing_diff', 'reach', 'vertical', 'agility', 'sprint',
        'pos_g', 'pos_f', 'pos_c', 'bg_college', 'bg_international', 'bg_high', 'bg_g']
COLLEGE = ['games', 'mpg'] + RATE_STATS
COMBINE = ['wingspan', 'reach', 'vertical', 'agility', 'sprint']

# weights on the last three college seasons, newest first
SEASON_WEIGHTS = {'last': [1, 0, 0], 'new': [5, 3, 2], 'old': [2, 3, 5]}

# what v4 uses, picked with ablation.py. only the season weighting helped, the rest stays switched off
OPTIONS = dict(seasons='new', schedule=False, shrink=False, age=False, position=False, flags=False)

AGE_REF = 20.0
MIN_MINUTES = 100


def load_college():
    seasons = pd.read_csv(os.path.join(FINAL, 'college_seasons.csv'))
    priors = pd.read_csv(os.path.join(FINAL, 'college_priors.csv'))
    with open(os.path.join(FINAL, 'college_adjust.json')) as f:
        adj = json.load(f)
    return seasons, priors, adj


def pooled(seasons, df, scheme):
    # add up the last three seasons with the weights, so the rates come out of one bigger sample
    s = seasons.merge(df[['person_id', 'college_season']].rename(columns={'college_season': 'last'}), on='person_id')
    s = s[s['college_season'] <= s['last']].copy()
    s['back'] = (s['last'] - s['college_season']).astype(int)
    w = SEASON_WEIGHTS[scheme]
    s['w'] = s['back'].map(lambda b: w[b] if b < 3 else 0)
    n_seasons = s.groupby('person_id').size()
    s = s[s['w'] > 0]
    cols = COUNTS + TEAM_COUNTS + ['games']
    tot = s[cols].multiply(s['w'], axis=0).groupby(s['person_id']).sum()
    wm = s['w'] * s['min']
    for c in ['sos', 'team_rating', 'back']:
        tot[c] = (s[c] * wm).groupby(s['person_id']).sum() / wm.groupby(s['person_id']).sum()
    # effective minutes and attempts, so heavier weights don't pretend to be more data
    for c in ['min', 'fga', 'fg3a', 'fta']:
        a = s['w'] * s[c]
        tot['n_' + c] = (a.groupby(s['person_id']).sum() ** 2 / (s['w'] * a).groupby(s['person_id']).sum()).fillna(0)
    tot['college_seasons'] = n_seasons
    # games is a count, put it back on the scale of one season
    tot['games'] = tot['games'] / s.groupby('person_id')['w'].sum() * s.groupby('person_id')['w'].max()
    return tot


# ---------- feature steps ----------
# each step gets the rates table and can change it in place, and returns any new model features.
# a new idea is a new function added to STEPS, then switched on in OPTIONS

def schedule_step(c):
    # credit a player for a tougher schedule, by how much each stat drops against better teams
    f = c['f']
    for stat, b in c['adj']['schedule_slope'].items():
        before = f[stat].copy()
        f[stat] = f[stat] - b * f['sos']
        ratio = (f[stat] / before).where(before > 0)
        for rate_stat, base in [('usg', 'fga40'), ('ast_pct', 'ast40'), ('orb_pct', 'orb40'),
                                ('drb_pct', 'drb40'), ('stl_pct', 'stl40'), ('blk_pct', 'blk40')]:
            if base == stat:
                f[rate_stat] = f[rate_stat] * ratio.fillna(1)
    f['trb40'] = f['orb40'] + f['drb40']
    return ['m_sos', 'm_team_rating']


def shrink_step(c):
    # small samples get pulled toward the average d1 player at his position that season
    f, prior = c['f'], c['prior']
    has = f['minutes'].notna()
    for stat in RATE_STATS:
        w = c['n'][stat].reindex(c['df']['person_id'].values).values
        w = w / (w + c['adj']['shrink_k'][stat])
        f[stat] = np.where(has, w * f[stat].fillna(prior[stat]) + (1 - w) * prior[stat], np.nan)
    return []


def age_step(c):
    # a year older is about a year more developed, so put everyone on a 20 year old's scale
    f = c['f']
    years = (c['df']['age'] - f['eff_age_back'] - AGE_REF).fillna(0)
    for stat in RATE_STATS:
        f[stat] = f[stat] - c['adj']['season_change'][stat] * years
    return []


def position_step(c):
    # relative to the average d1 player at his position that season, which also takes out the era
    for stat in RATE_STATS:
        c['f'][stat] = c['f'][stat] - c['prior'][stat]
    return []


def flags_step(c):
    # skipping the combine says something on its own
    df = c['df']
    df['no_combine'] = df[COMBINE].isna().all(axis=1).astype(int)
    df['no_athletic'] = (df['vertical'].isna() & df['wingspan'].notna()).astype(int)
    return ['no_combine', 'no_athletic', 'm_college_seasons']


STEPS = [('schedule', schedule_step), ('shrink', shrink_step), ('age', age_step),
         ('position', position_step), ('flags', flags_step)]


def make_features(df, options=None):
    o = dict(OPTIONS, **(options or {}))
    seasons, priors, adj = load_college()
    df = df.copy()
    tot = pooled(seasons, df[df['espn_id'].notna()], o['seasons'])
    f = rates(tot)
    n = {c: tot['n_' + EXPOSURE.get(c, 'min').replace('tsa', 'fga')] for c in RATE_STATS}
    for c in ['sos', 'team_rating', 'college_seasons']:
        f[c] = tot[c]
    f['eff_age_back'] = tot['back']
    f['minutes'] = tot['min']
    f = f.reindex(df['person_id'].values)
    f.index = df.index

    key = pd.DataFrame({'college_season': df['college_season'], 'pos': df['espn_pos'].apply(pos_group)}, index=df.index)
    prior = key.merge(priors, on=['college_season', 'pos'], how='left')
    prior.index = df.index

    context = {'df': df, 'f': f, 'prior': prior, 'n': n, 'adj': adj}
    extra = []
    for name, step in STEPS:
        if o.get(name):
            extra += step(context)

    for c in COLLEGE + ['sos', 'team_rating', 'college_seasons']:
        df['m_' + c] = f[c]
    feats = BASE + ['m_' + c for c in COLLEGE] + extra
    if 'm_sos' not in extra:
        feats += ['team_margin', 'team_srs']

    # very low minute college seasons are mostly noise, so treat them like no stats at all
    low = f['minutes'].fillna(0) < MIN_MINUTES
    college_cols = [c for c in feats if c.startswith('m_') or c in ['team_margin', 'team_srs']]
    df.loc[low, college_cols] = np.nan
    return df, feats
