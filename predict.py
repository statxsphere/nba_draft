import os
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd

from train import TARGETS, FLOORS, FIRST_TEST, load_data, to_vorp, predict_blend, blend_shap
from build_dataset import PEAK_SCALE, IMM_SCALE, LAST_NBA_SEASON

ROOT = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(ROOT, 'data', 'final')
MODELS = os.path.join(ROOT, 'models')
SITE_DATA = os.path.join(ROOT, 'docs', 'data')

# the two vorp ratings are scaled to -10..10, the per possession targets stay in points per 100
SCALES = {'peak': PEAK_SCALE, 'imm': IMM_SCALE}
NAMES = {'peak': 'Peak Impact', 'imm': 'Immediate Impact', 'bpm': 'Peak BPM', 'rapm': 'Peak RAPM',
         'imm_rapm': 'Immediate RAPM'}

# v3 as it was published, before the target fix and the nested tuning
V3_PUBLISHED = {'peak': {'model': 0.374, 'draft_order': 0.447, 'model_pick': 0.437, 'top10': 3.5, 'draft_top10': 4.0,
                         'coverage': 0.83, 'classes': 11},
                'imm': {'model': 0.289, 'draft_order': 0.059, 'model_pick': 0.299, 'top10': 3.3, 'draft_top10': 3.9,
                        'coverage': 0.74, 'classes': 15}}

COMP_FEATURES = ['age', 'height', 'weight', 'wing_diff', 'reach', 'vertical', 'pos_g', 'pos_f', 'pos_c',
                 'pts40', 'trb40', 'ast40', 'stl40', 'blk40', 'tov40', 'fg3a40', 'fta40', 'ts', 'fg3_pct',
                 'ft_pct', 'usg', 'ast_pct', 'orb_pct', 'drb_pct', 'stl_pct', 'blk_pct', 'team_srs']

# plain words for every feature: label, how to print the value
LABELS = {
    'age': ('Age on draft night', '{:.1f}'),
    'height': ('Height', 'height'),
    'weight': ('Weight', '{:.0f} lb'),
    'wingspan': ('Wingspan', 'height'),
    'wing_diff': ('Wingspan minus height', '{:+.1f} in'),
    'reach': ('Standing reach', 'height'),
    'vertical': ('Max vertical', '{:.1f} in'),
    'agility': ('Lane agility time', '{:.2f} s'),
    'sprint': ('Three-quarter sprint', '{:.2f} s'),
    'pos_g': ('Plays guard', 'yesno'),
    'pos_f': ('Plays forward', 'yesno'),
    'pos_c': ('Plays center', 'yesno'),
    'bg_college': ('Came from college', 'yesno'),
    'bg_international': ('Came from overseas', 'yesno'),
    'bg_high': ('Came from high school', 'yesno'),
    'bg_g': ('Came from the G League / OTE', 'yesno'),
    'games': ('College games', '{:.0f}'),
    'mpg': ('Minutes per game', '{:.1f}'),
    'pts40': ('Points per 40', '{:.1f}'),
    'trb40': ('Rebounds per 40', '{:.1f}'),
    'orb40': ('Offensive rebounds per 40', '{:.1f}'),
    'drb40': ('Defensive rebounds per 40', '{:.1f}'),
    'ast40': ('Assists per 40', '{:.1f}'),
    'stl40': ('Steals per 40', '{:.1f}'),
    'blk40': ('Blocks per 40', '{:.1f}'),
    'tov40': ('Turnovers per 40', '{:.1f}'),
    'pf40': ('Fouls per 40', '{:.1f}'),
    'fga40': ('Shot attempts per 40', '{:.1f}'),
    'fg3a40': ('Threes attempted per 40', '{:.1f}'),
    'fta40': ('Free throws attempted per 40', '{:.1f}'),
    'ts': ('True shooting', 'pct'),
    'efg': ('Effective FG%', 'pct'),
    'fg3_pct': ('Three-point %', 'pct'),
    'ft_pct': ('Free throw %', 'pct'),
    'fg3a_rate': ('Share of shots from three', 'pct'),
    'fta_rate': ('Free throw rate', '{:.2f}'),
    'usg': ('Usage rate', '{:.1f}%'),
    'ast_pct': ('Assist rate', '{:.1f}%'),
    'orb_pct': ('Offensive rebound rate', '{:.1f}%'),
    'drb_pct': ('Defensive rebound rate', '{:.1f}%'),
    'stl_pct': ('Steal rate', '{:.1f}%'),
    'blk_pct': ('Block rate', '{:.1f}%'),
    'tov_pct': ('Turnover rate', '{:.1f}%'),
    'ast_tov': ('Assist to turnover', '{:.2f}'),
    'rim_rate': ('Share of shots at the rim', 'pct'),
    'mid_rate': ('Share of shots from mid range', 'pct'),
    'dunk_rate': ('Share of shots that are dunks', 'pct'),
    'rim_pct': ('FG% at the rim', 'pct'),
    'mid_pct': ('FG% from mid range', 'pct'),
    'assisted_share': ('Share of makes assisted', 'pct'),
    'team_margin': ('Team point margin', '{:+.1f}'),
    'team_srs': ('Team strength (schedule adjusted)', '{:+.1f}'),
}


def label(feature):
    return LABELS[feature[2:] if feature.startswith('m_') else feature]


def show_value(feature, value):
    if pd.isna(value):
        return 'not available'
    fmt = label(feature)[1]
    if fmt == 'height':
        return f"{int(value // 12)}'{value % 12:.1f}\""
    if fmt == 'yesno':
        return 'yes' if value >= 0.5 else 'no'
    if fmt == 'pct':
        return f'{100 * value:.1f}%'
    return fmt.format(value)


def rating(v, key):
    if key not in SCALES:
        return float(np.round(v, 2))
    return float(np.clip(10 * v / SCALES[key], -10, 10).round(2))


def num(x, digits=3):
    if x is None or pd.isna(x):
        return None
    return round(float(x), digits)


def latest_predictions(df, key):
    # classes the final model never saw get their numbers from it, older ones keep their backtest numbers
    fitted = joblib.load(os.path.join(MODELS, f'{key}.pkl'))
    new = df[df['draft_year'] > fitted['last_class']].copy()
    new['score'] = predict_blend(fitted, new)
    new['model'], new['low'], new['high'] = to_vorp(new['score'].values, fitted['map'])
    new['target'] = key
    values = blend_shap(fitted, new)
    X = new[fitted['feats']]
    rows = []
    for i in range(len(X)):
        for j in np.argsort(-np.abs(values[i]))[:6]:
            rows.append({'person_id': new['person_id'].iloc[i], 'target': key, 'feature': X.columns[j],
                         'value': X.iloc[i, j], 'shap': values[i][j]})
    return new[['person_id', 'target', 'score', 'model', 'low', 'high']], pd.DataFrame(rows), fitted['last_class']


def comp_table(full):
    z = full[COMP_FEATURES].astype(float)
    pool = full['peak_complete'] == True
    mean = z[pool].mean()
    std = z[pool].std()
    return (z - mean) / std


def find_comps(full, z, i, n=10):
    p = full.iloc[i]
    a = z.iloc[i].values
    has = ~np.isnan(a)
    pool = (full['peak_complete'] == True) & (full['draft_year'] < p['draft_year'])
    if p['has_college_stats'] == 1:
        pool &= full['has_college_stats'] == 1
    else:
        pool &= full['background'] != 'college'
    idx = np.where(pool.values)[0]
    if len(idx) == 0 or has.sum() < 3:
        return []
    B = z.iloc[idx].values
    both = has & ~np.isnan(B)
    a0 = np.where(has, a, 0)
    B0 = np.where(np.isnan(B), 0, B)
    top = (B0 * a0).sum(axis=1)
    bottom = np.sqrt(((a0 ** 2) * both).sum(axis=1) * ((B0 ** 2) * both).sum(axis=1))
    sim = np.where(bottom > 0, top / np.where(bottom > 0, bottom, 1), -1)
    # need most of his features to line up, or the similarity means nothing
    sim = np.where(both.sum(axis=1) >= 0.75 * has.sum(), sim, -1)
    best = np.argsort(-sim)[:n]
    return [(idx[b], sim[b]) for b in best if sim[b] > 0]


def show_comps(comps):
    # the five closest, plus the ceiling and floor out of the closest ten
    if not comps:
        return []
    out = comps[:5]
    for c in [max(comps, key=lambda c: c['peak']), min(comps, key=lambda c: c['peak'])]:
        if c not in out:
            out.append(c)
    return out


def drivers_for(drivers, pid, key):
    d = drivers[(drivers['person_id'] == pid) & (drivers['target'] == key)]
    out = []
    for _, r in d.sort_values('shap', key=abs, ascending=False).head(5).iterrows():
        out.append({'label': label(r['feature'])[0], 'value': show_value(r['feature'], r['value']),
                    'feature': r['feature'], 'pctl': num(100 * r['shap'], 1)})
    return out


def season_line(seasons, pid):
    s = seasons[seasons['person_id'] == pid].sort_values('k')
    return [{'season': f"{int(r['season']) - 1}-{str(int(r['season']))[-2:]}", 'g': int(r['g']),
             'mp': int(r['mp']), 'vorp': num(r['vorp'], 1), 'bpm': num(r['bpm'], 1), 'ws': num(r['ws'], 1)}
            for _, r in s.iterrows()]


def records(t):
    # json has no nan, so empty cells become null
    t = t.round(3).astype(object)
    return t.where(t.notna(), None).to_dict('records')


def mean_of(g, col):
    return num(g[col].mean()) if col in g and g[col].notna().any() else None


def backtest_summary(metrics, back, importance):
    summary = {}
    for key in TARGETS:
        m = metrics[metrics['target'] == key]
        p = back[back['target'] == key].copy()
        years = sorted(m['year'].unique().tolist())
        by_model = {}
        for model, g in m.groupby('model'):
            g = g.sort_values('year')
            by_model[model] = {'spearman': [num(x) for x in g['spearman']], 'top10': g['top10'].tolist(),
                               'ndcg10': [num(x) for x in g['ndcg10']],
                               'mean_spearman': num(g['spearman'].mean()), 'mean_top10': num(g['top10'].mean(), 2),
                               'mean_ndcg10': num(g['ndcg10'].mean()), 'coverage': mean_of(g, 'coverage'),
                               'width': mean_of(g, 'width'),
                               'coverage_by_year': [num(x) for x in g['coverage']] if g['coverage'].notna().any() else None}
        model_s = m[m['model'] == 'model'].set_index('year')['spearman']
        draft_s = m[m['model'] == 'draft_order'].set_index('year')['spearman']
        pick_s = m[m['model'] == 'model_pick'].set_index('year')['spearman']
        p['bin'] = pd.qcut(p['model'].rank(method='first'), 10, labels=False)
        calib = p.groupby('bin').agg(pred=('model', 'mean'), actual=('actual', 'mean'), n=('model', 'size'),
                                      low=('low', 'mean'), high=('high', 'mean')).reset_index()
        first_round = []
        for year in years:
            t = p[(p['draft_year'] == year) & (p['pick'] <= 30)]
            first_round.append({'year': int(year),
                                'model': num(t[['model', 'actual']].corr('spearman').iloc[0, 1]),
                                'draft_order': num(t[['pick', 'actual']].corr('spearman').iloc[0, 1] * -1)})
        summary[key] = {
            'name': NAMES[key], 'floor': FLOORS[key], 'rating_scale': SCALES.get(key),
            'years': [int(y) for y in years], 'models': by_model,
            'model_beats_draft': int((model_s > draft_s).sum()), 'pick_beats_draft': int((pick_s > draft_s).sum()),
            'classes': len(years),
            'calibration': [{k: num(v) for k, v in r.items()} for r in calib.to_dict('records')],
            'first_round': first_round,
            'scatter': [{'name': r['name'], 'year': int(r['draft_year']), 'pick': int(r['pick']),
                         'pred': num(r['model']), 'actual': num(r['actual'])} for _, r in p.iterrows()],
            'importance': [{'feature': r['feature'], 'label': label(r['feature'])[0], 'shap': num(100 * r['mean_abs_shap'], 2)}
                           for _, r in importance[importance['target'] == key].head(15).iterrows()],
        }
    return summary


def main():
    os.makedirs(SITE_DATA, exist_ok=True)
    df, feats = load_data()
    full = pd.read_csv(os.path.join(FINAL, 'draft_players.csv'))
    full.loc[full['games'].fillna(0) * full['mpg'].fillna(0) < 100, COMP_FEATURES[9:-1]] = np.nan
    seasons = pd.read_csv(os.path.join(FINAL, 'nba_seasons.csv'))
    back = pd.read_csv(os.path.join(FINAL, 'backtest_predictions.csv'))
    back_drivers = pd.read_csv(os.path.join(FINAL, 'backtest_drivers.csv'))
    metrics = pd.read_csv(os.path.join(FINAL, 'backtest_metrics.csv'))
    importance = pd.read_csv(os.path.join(FINAL, 'feature_importance.csv'))

    preds, drivers, last_class = [back[['person_id', 'target', 'score', 'model', 'low', 'high']]], [back_drivers], {}
    for key in TARGETS:
        p, d, last_class[key] = latest_predictions(df, key)
        preds.append(p)
        drivers.append(d)
    preds = pd.concat(preds)
    drivers = pd.concat(drivers)

    board = df[df['draft_year'] >= FIRST_TEST].copy()
    for key in TARGETS:
        p = preds[preds['target'] == key].drop(columns='target')
        p.columns = ['person_id'] + [f'{key}_{c}' for c in p.columns[1:]]
        board = board.merge(p, on='person_id', how='left')
    board['model_rank'] = board.groupby('draft_year')['peak_score'].rank(ascending=False, method='first').astype(int)
    board['imm_rank'] = board.groupby('draft_year')['imm_score'].rank(ascending=False, method='first').astype(int)

    z = comp_table(full)
    full_index = {pid: i for i, pid in enumerate(full['person_id'])}

    print('writing the boards..')
    classes = []
    for year, group in board.groupby('draft_year'):
        players = []
        for _, p in group.sort_values('pick').iterrows():
            comps = []
            for j, sim in find_comps(full, z, full_index[p['person_id']]):
                c = full.iloc[j]
                comps.append({'name': c['name'], 'year': int(c['draft_year']), 'pick': int(c['pick']),
                              'sim': num(sim, 3), 'peak': num(c['peak_rating'], 2), 'imm': num(c['imm_rating'], 2),
                              'line': [x['vorp'] for x in season_line(seasons, c['person_id'])]})
            pred = {}
            for key in [k for k in TARGETS if k != 'imm_rapm']:
                pred[key] = {'score': num(p[f'{key}_score']), 'vorp': num(p[f'{key}_model']), 'low': num(p[f'{key}_low']),
                             'high': num(p[f'{key}_high']), 'rating': rating(p[f'{key}_model'], key),
                             'rating_low': rating(p[f'{key}_low'], key), 'rating_high': rating(p[f'{key}_high'], key)}
                if key in ['peak', 'imm']:
                    pred[key]['drivers'] = drivers_for(drivers, p['person_id'], key)
            players.append({
                'id': int(p['person_id']), 'name': p['name'], 'pick': int(p['pick']), 'round': int(p['round']),
                'team': p['team'], 'from': p['organization'] if pd.notna(p['organization']) else '', 'background': p['background'],
                'position': p['position'] if pd.notna(p['position']) else '', 'age': num(p['age'], 1),
                'height': show_value('height', p['height']) if pd.notna(p['height']) else None,
                'wingspan': show_value('wingspan', p['wingspan']) if pd.notna(p['wingspan']) else None,
                'weight': num(p['weight'], 0),
                'college': None if pd.isna(p['pts40']) else {
                    'season': f"{int(p['college_season']) - 1}-{str(int(p['college_season']))[-2:]}",
                    'team': p['espn_team'], 'g': int(p['games']), 'mpg': num(p['mpg'], 1),
                    'pts40': num(p['pts40'], 1), 'trb40': num(p['trb40'], 1), 'ast40': num(p['ast40'], 1),
                    'stl40': num(p['stl40'], 1), 'blk40': num(p['blk40'], 1), 'ts': num(p['ts'], 3),
                    'fg3_pct': num(p['fg3_pct'], 3), 'ft_pct': num(p['ft_pct'], 3), 'usg': num(p['usg'], 1),
                    'rim_rate': num(p.get('m_rim_rate'), 3), 'rim_pct': num(p.get('m_rim_pct'), 3)},
                'model_rank': int(p['model_rank']), 'imm_rank': int(p['imm_rank']), 'pred': pred,
                'actual': {'peak': num(p['peak_vorp']), 'imm': num(p['imm_vorp']),
                           'peak_rating': num(p['peak_rating'], 2), 'imm_rating': num(p['imm_rating'], 2),
                           'peak_so_far': num(p['peak_vorp_so_far']), 'imm_so_far': num(p['imm_vorp_so_far']),
                           'peak_complete': bool(p['peak_complete']), 'imm_complete': bool(p['imm_complete']),
                           'bpm': num(p['peak_bpm']), 'bpm_so_far': num(p['peak_bpm_so_far']),
                           'rapm': num(p.get('peak_rapm'))},
                'seasons': season_line(seasons, p['person_id']),
                'comps': show_comps(comps),
                'ceiling': max(comps, key=lambda c: c['peak'])['name'] if comps else None,
                'floor': min(comps, key=lambda c: c['peak'])['name'] if comps else None,
            })
        with open(os.path.join(SITE_DATA, f'board_{year}.json'), 'w', encoding='utf-8', newline='\n') as f:
            json.dump(players, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        status = 'complete' if year <= last_class['peak'] else ('partial' if year < LAST_NBA_SEASON else 'none')
        classes.append({'year': int(year), 'players': len(players), 'outcomes': status})
        print(f'  {year}: {len(players)} players')

    print('writing the backtest..')
    summary = {'targets': backtest_summary(metrics, back, importance), 'v3_published': V3_PUBLISHED}
    summary['audit'] = records(pd.read_csv(os.path.join(FINAL, 'target_audit.csv')))
    summary['ablation'] = records(pd.read_csv(os.path.join(FINAL, 'ablation.csv')))
    check = os.path.join(FINAL, 'rapm_check.json')
    if os.path.exists(check):
        with open(check) as f:
            summary['rapm_check'] = json.load(f)
    with open(os.path.join(SITE_DATA, 'backtest.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

    meta = {'updated': date.today().isoformat(), 'classes': classes, 'scales': SCALES, 'targets': list(TARGETS),
            'names': {k: NAMES[k] for k in TARGETS}, 'last_complete': last_class,
            'last_nba_season': f'{LAST_NBA_SEASON - 1}-{str(LAST_NBA_SEASON)[-2:]}',
            'players': int(len(full)), 'train_players': {k: int(df[TARGETS[k]].notna().sum()) for k in TARGETS}}
    with open(os.path.join(SITE_DATA, 'meta.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    cols = ['person_id', 'name', 'draft_year', 'pick', 'background', 'model_rank', 'imm_rank']
    for key in TARGETS:
        cols += [f'{key}_score', f'{key}_model', f'{key}_low', f'{key}_high']
    out = board[cols + ['peak_vorp', 'imm_vorp', 'peak_bpm', 'peak_vorp_so_far']].copy()
    for key in SCALES:
        for c in ['model', 'low', 'high']:
            out[f'{key}_{c}_rating'] = (10 * out[f'{key}_{c}'] / SCALES[key]).clip(-10, 10).round(2)
    out.round(4).to_csv(os.path.join(FINAL, 'projections.csv'), index=False)

    latest = board[board['draft_year'] == board['draft_year'].max()].sort_values('peak_score', ascending=False)
    print(f"\ntop of the {latest['draft_year'].iloc[0]} class by projected peak:")
    for _, r in latest.head(10).iterrows():
        print(f"  {r['name']:<24} pick {r['pick']:>2}  peak {rating(r['peak_model'], 'peak'):>5}"
              f"  ({rating(r['peak_low'], 'peak')} to {rating(r['peak_high'], 'peak')})  immediate {rating(r['imm_model'], 'imm')}")
    print('Done!')


if __name__ == '__main__':
    main()
