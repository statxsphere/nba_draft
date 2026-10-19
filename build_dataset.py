import os
import re
import json
import time
import difflib
import unicodedata

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, 'data', 'raw')
FINAL = os.path.join(ROOT, 'data', 'final')

FIRST_DRAFT = 2000
LAST_DRAFT = 2026
FIRST_COLLEGE_SEASON = 2005   # the espn box scores are only complete from 2004-05
LAST_NBA_SEASON = 2026        # 2025-26, the last season in the bbref snapshot

PEAK_SEASONS = 6              # peak = best 3 season run inside the first 6 seasons
PEAK_SCALE = 4.0              # 4 vorp a season over the run = 10
IMM_SCALE = 2.0               # 2 vorp a season over the first two = 10
REPLACEMENT_BPM = -2.0        # bpm of a replacement level player, what vorp is measured from
BPM_PRIOR_MIN = 750           # minutes of replacement level added before taking a player's bpm

HEADERS = {'User-Agent': 'nba_draft data script (github.com/statxsphere/nba_draft)'}

BBREF_DATA = 'https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/Data/'
ESPN_BOX = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_player_boxscores/'

# names that espn and nba.com spell too differently to match automatically
NAME_FIXES = {
    'Bub Carrington': 'Carlton Carrington',
}

BBREF_FIXES = {
    'Ha Ha': 'Ha Seung-Jin',
    'Sun Sun': 'Sun Yue',
}

SCHOOL_FIXES = {
    'brigham young': 'byu',
    'connecticut': 'uconn',
    'louisiana state': 'lsu',
    'southern california': 'usc',
    'virginia commonwealth': 'vcu',
    'central florida': 'ucf',
    'nevada las vegas': 'unlv',
    'texas christian': 'tcu',
    'southern methodist': 'smu',
    'california los angeles': 'ucla',
    'alabama birmingham': 'uab',
    'texas el paso': 'utep',
    'mississippi': 'ole miss',
    'massachusetts': 'umass',
    'pittsburgh': 'pitt',
    'miami fl': 'miami',
    'north carolina state': 'nc state',
    'texas a m': 'texas a&m',
}


def get_file(url, name):
    path = os.path.join(RAW, name)
    if os.path.exists(path):
        return path
    print(f'downloading {url}')
    r = requests.get(url, headers=HEADERS, timeout=300)
    r.raise_for_status()
    with open(path, 'wb') as f:
        f.write(r.content)
    time.sleep(2)
    return path


def nba_call(name, func, required=True):
    # every stats.nba.com response is saved, so a rerun never asks twice
    path = os.path.join(RAW, 'nba_api', name + '.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    for attempt in range(3):
        try:
            time.sleep(1.5)
            df = func()
            df.to_csv(path, index=False)
            return df
        except KeyError:
            # nba.com has no record for this id
            break
        except Exception as e:
            print(f'{name} failed ({e}), waiting..')
            time.sleep(10)
    if required:
        raise RuntimeError(f'could not get {name}')
    # remember the miss so a rerun doesn't keep asking
    pd.DataFrame(columns=['failed']).to_csv(path, index=False)
    return None


def clean_name(x):
    x = unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode()
    x = x.lower().replace('.', '').replace("'", '').replace('-', ' ')
    x = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', x)
    return ' '.join(x.split())


def clean_school(x):
    x = clean_name(x).replace('&', ' ').replace('(', ' ').replace(')', ' ')
    x = x.replace('university', '').replace('college', '').replace('of ', '')
    x = ' '.join(x.split())
    return SCHOOL_FIXES.get(x, x)


def background(org, org_type):
    org = str(org)
    if org_type == 'College/University':
        return 'college'
    if org_type == 'High School' or 'Academy' in org or 'Prep' in org or 'High School' in org:
        return 'high school'
    if 'G League' in org or 'Ignite' in org or 'Overtime Elite' in org or 'City Reapers' in org:
        return 'g league / ote'
    return 'international'


def height_to_inches(x):
    try:
        feet, inches = str(x).split('-')
        return int(feet) * 12 + float(inches)
    except Exception:
        return np.nan


# ---------- draft history and measurements (nba_api) ----------

def get_draft():
    from nba_api.stats.endpoints import drafthistory
    df = nba_call('draft_history', lambda: drafthistory.DraftHistory(league_id='00', timeout=60).get_data_frames()[0])
    df['SEASON'] = df['SEASON'].astype(int)
    df = df[(df['SEASON'] >= FIRST_DRAFT) & (df['SEASON'] <= LAST_DRAFT)].copy()
    df = df.rename(columns={'PERSON_ID': 'person_id', 'PLAYER_NAME': 'name', 'SEASON': 'draft_year',
                            'ROUND_NUMBER': 'round', 'OVERALL_PICK': 'pick', 'TEAM_ABBREVIATION': 'team',
                            'ORGANIZATION': 'organization', 'ORGANIZATION_TYPE': 'organization_type'})
    df['background'] = [background(o, t) for o, t in zip(df['organization'], df['organization_type'])]
    df = df[['person_id', 'name', 'draft_year', 'round', 'pick', 'team', 'organization', 'background']]
    return df.sort_values(['draft_year', 'pick']).reset_index(drop=True)


def get_combine():
    from nba_api.stats.endpoints import draftcombinestats
    frames = []
    for year in range(FIRST_DRAFT, LAST_DRAFT + 1):
        season = f'{year}-{str(year + 1)[-2:]}'
        df = nba_call(f'combine_{year}', lambda: draftcombinestats.DraftCombineStats(
            league_id='00', season_all_time=season, timeout=60).get_data_frames()[0])
        frames.append(df)
    df = pd.concat(frames)
    df = df.rename(columns={'PLAYER_ID': 'person_id', 'POSITION': 'combine_pos',
                            'HEIGHT_WO_SHOES': 'combine_height', 'WEIGHT': 'combine_weight',
                            'WINGSPAN': 'wingspan', 'STANDING_REACH': 'reach',
                            'MAX_VERTICAL_LEAP': 'vertical', 'LANE_AGILITY_TIME': 'agility',
                            'THREE_QUARTER_SPRINT': 'sprint', 'BODY_FAT_PCT': 'body_fat'})
    cols = ['person_id', 'combine_pos', 'combine_height', 'combine_weight', 'wingspan', 'reach',
            'vertical', 'agility', 'sprint', 'body_fat']
    df = df[cols].drop_duplicates('person_id', keep='last')
    for c in cols[2:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def get_player_info(person_ids):
    from nba_api.stats.endpoints import commonplayerinfo
    rows = []
    for i, pid in enumerate(person_ids):
        df = nba_call(f'player_{pid}', lambda: commonplayerinfo.CommonPlayerInfo(
            player_id=pid, timeout=60).get_data_frames()[0], required=False)
        if df is not None and len(df):
            rows.append(df.iloc[0])
        if i % 25 == 0:
            print(f'player info {i}/{len(person_ids)}')
    df = pd.DataFrame(rows)
    df = df.rename(columns={'PERSON_ID': 'person_id', 'BIRTHDATE': 'nba_birth', 'POSITION': 'nba_pos'})
    df['nba_height'] = df['HEIGHT'].apply(height_to_inches)
    df['nba_weight'] = pd.to_numeric(df['WEIGHT'], errors='coerce')
    return df[['person_id', 'nba_birth', 'nba_pos', 'nba_height', 'nba_weight']]


# ---------- nba outcomes (basketball reference tables) ----------

def get_bbref():
    adv = pd.read_csv(get_file(BBREF_DATA + 'Advanced.csv', 'bbref_advanced.csv'))
    picks = pd.read_csv(get_file(BBREF_DATA + 'Draft%20Pick%20History.csv', 'bbref_draft_picks.csv'))
    info = pd.read_csv(get_file(BBREF_DATA + 'Player%20Career%20Info.csv', 'bbref_career_info.csv'))

    adv = adv[adv['lg'] == 'NBA']
    # traded players have a row per team plus a total row, keep only the total
    adv['is_total'] = adv['team'].astype(str).str.match(r'^(TOT|\dTM)$')
    adv = adv.sort_values('is_total', ascending=False).drop_duplicates(['player_id', 'season'])
    adv = adv[['player_id', 'season', 'age', 'g', 'mp', 'bpm', 'ws', 'vorp']]

    picks = picks[(picks['lg'] == 'NBA') & (picks['season'] >= FIRST_DRAFT)]
    picks = picks[['season', 'overall_pick', 'player', 'player_id']].rename(
        columns={'season': 'draft_year', 'overall_pick': 'pick', 'player': 'bbref_name'})
    info = info[['player_id', 'pos', 'ht_in_in', 'wt', 'birth_date']].rename(
        columns={'pos': 'bbref_pos', 'ht_in_in': 'bbref_height', 'wt': 'bbref_weight', 'birth_date': 'bbref_birth'})
    return adv, picks, info


def link_bbref(df, picks):
    # nba.com and bbref number a few forfeited picks differently, so match on name inside each class
    ids = []
    for _, p in df.iterrows():
        pool = picks[picks['draft_year'] == p['draft_year']]
        keys = pool['bbref_name'].apply(clean_name).tolist()
        found = None
        key = clean_name(BBREF_FIXES.get(p['name'], p['name']))
        if key in keys:
            found = pool.iloc[keys.index(key)]['player_id']
        else:
            close = difflib.get_close_matches(key, keys, n=1, cutoff=0.75)
            if close:
                found = pool.iloc[keys.index(close[0])]['player_id']
            else:
                near = pool[(pool['pick'] - p['pick']).abs() <= 2]
                last = key.split(' ')[-1]
                near = near[near['bbref_name'].apply(clean_name).str.split(' ').str[-1] == last]
                if len(near) == 1:
                    found = near.iloc[0]['player_id']
        ids.append(found)
    df['player_id'] = ids
    print(f"{df['player_id'].notna().sum()} of {len(df[df['draft_year'] <= picks['draft_year'].max()])} picks linked to bbref")
    unlinked = df[df['player_id'].isna() & (df['draft_year'] <= picks['draft_year'].max())]
    for _, r in unlinked.iterrows():
        print(f"  not on bbref: {r['draft_year']} #{r['pick']} {r['name']}")
    return df


def nba_outcomes(df, adv):
    seasons = []
    for _, p in df.iterrows():
        n = min(LAST_NBA_SEASON - p['draft_year'], 10)
        played = adv[adv['player_id'] == p['player_id']].set_index('season') if pd.notna(p['player_id']) else None
        for k in range(1, n + 1):
            season = p['draft_year'] + k
            row = {'person_id': p['person_id'], 'draft_year': p['draft_year'], 'k': k, 'season': season,
                   'g': 0, 'mp': 0, 'bpm': np.nan, 'ws': 0.0, 'vorp': 0.0}
            if played is not None and season in played.index:
                s = played.loc[season]
                row.update({'g': s['g'], 'mp': s['mp'], 'bpm': s['bpm'], 'ws': s['ws'], 'vorp': s['vorp']})
            seasons.append(row)
    return pd.DataFrame(seasons)


def best_run(v):
    if len(v) < 3:
        return np.mean(v) if len(v) else np.nan
    return max(np.mean(v[i:i + 3]) for i in range(len(v) - 2))


def best_bpm_run(bpm, mp):
    # per possession: best 3 season run of bpm, with the minutes pooled and 750 minutes of
    # replacement level (-2) added, so a short stint can't look like a star
    if len(bpm) == 0:
        return np.nan
    runs = []
    for i in range(max(len(bpm) - 2, 1)):
        b, m = bpm[i:i + 3], mp[i:i + 3]
        runs.append((np.sum(b * m) + BPM_PRIOR_MIN * REPLACEMENT_BPM) / (np.sum(m) + BPM_PRIOR_MIN))
    return max(max(runs), REPLACEMENT_BPM)


def add_targets(df, seasons):
    # a season below replacement counts as replacement, the same as a season he didn't play.
    # without this a pick who never plays (0) ranks above a rotation player with -0.5
    seasons['vorp_floor'] = seasons['vorp'].clip(lower=0)
    rows = []
    for _, p in df.iterrows():
        s = seasons[seasons['person_id'] == p['person_id']].sort_values('k')
        raw, v = s['vorp'].values, s['vorp_floor'].values
        bpm, mp = s['bpm'].fillna(REPLACEMENT_BPM).values, s['mp'].values
        full = len(v) >= PEAK_SEASONS
        r = {'peak_vorp': best_run(v[:PEAK_SEASONS]) if full else np.nan,
             'imm_vorp': np.mean(v[:2]) if len(v) >= 2 else np.nan,
             'peak_vorp_so_far': best_run(v[:PEAK_SEASONS]) if len(v) else np.nan,
             'imm_vorp_so_far': np.mean(v[:2]) if len(v) else np.nan,
             # the rookie year is left out here, it says much less about later seasons than year two does
             'peak_bpm': best_bpm_run(bpm[1:PEAK_SEASONS], mp[1:PEAK_SEASONS]) if full else np.nan,
             'peak_bpm_so_far': best_bpm_run(bpm[1:PEAK_SEASONS], mp[1:PEAK_SEASONS]) if len(v) >= 2 else np.nan,
             # the v3 versions, before the floor, kept so the change can be checked
             'peak_vorp_v3': best_run(raw[:PEAK_SEASONS]) if full else np.nan,
             'imm_vorp_v3': np.mean(raw[:2]) if len(raw) >= 2 else np.nan}
        rows.append(r)
    t = pd.DataFrame(rows, index=df.index)
    for c in t.columns:
        df[c] = t[c]
    df['peak_complete'] = df['peak_vorp'].notna()
    df['imm_complete'] = df['imm_vorp'].notna()
    df['peak_rating'] = (10 * df['peak_vorp'] / PEAK_SCALE).clip(-10, 10).round(2)
    df['imm_rating'] = (10 * df['imm_vorp'] / IMM_SCALE).clip(-10, 10).round(2)
    return df


# ---------- college stats (espn box scores via hoopR) ----------

COUNTS = ['min', 'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 'orb', 'drb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts']
TEAM_COUNTS = ['tm_min', 'tm_fgm', 'tm_fga', 'tm_fta', 'tm_tov', 'tm_orb', 'tm_drb',
               'op_fga', 'op_fg3a', 'op_fta', 'op_orb', 'op_drb', 'op_tov']
PER40 = ['pts', 'orb', 'drb', 'ast', 'stl', 'blk', 'tov', 'pf', 'fga', 'fg3a', 'fta']
# stats that get the schedule adjustment, with what each game is weighted by
GAME_STATS = {c + '40': 'min' for c in PER40}
GAME_STATS.update({'ts': 'tsa', 'fg3_pct': 'fg3a', 'ft_pct': 'fta'})


def team_ratings(box):
    # ridge regression on every game's margin: margin = home edge + team rating - opponent rating
    from scipy import sparse
    from sklearn.linear_model import Ridge
    games = box[box['home_away'] == 'home'].drop_duplicates('game_id')
    teams = sorted(set(box['team_id']) | set(box['opponent_team_id']))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(games)
    rows = np.r_[np.arange(n), np.arange(n)]
    cols = np.r_[games['team_id'].map(idx).values, games['opponent_team_id'].map(idx).values]
    vals = np.r_[np.ones(n), -np.ones(n)]
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(teams)))
    margin = (games['team_score'] - games['opponent_team_score']).clip(-30, 30).astype(float)
    r = Ridge(alpha=5.0).fit(X, margin)
    ratings = pd.Series(r.coef_, index=teams)
    return ratings - ratings.mean()


def schedule_slopes(box):
    # how much each stat moves against a stronger opponent, inside the same player's season
    g = box[box['min'] >= 10].copy()
    g['tsa'] = g['fga'] + 0.44 * g['fta']
    for c in PER40:
        g[c + '40'] = 40 * g[c] / g['min']
    g['ts'] = g['pts'] / (2 * g['tsa'])
    g['fg3_pct'] = g['fg3m'] / g['fg3a']
    g['ft_pct'] = g['ftm'] / g['fta']
    sums = {}
    for stat, w in GAME_STATS.items():
        d = g[(g[w] > 0) & g[stat].notna()][['athlete_id', stat, w, 'opp_rating']].copy()
        d['wx'] = d[w] * d[stat]
        d['wr'] = d[w] * d['opp_rating']
        t = d.groupby('athlete_id')[[w, 'wx', 'wr']].transform('sum')
        x = d[stat] - t['wx'] / t[w]
        r = d['opp_rating'] - t['wr'] / t[w]
        sums[stat] = [float((d[w] * x * r).sum()), float((d[w] * r * r).sum())]
    return sums


def college_season(year):
    path = get_file(ESPN_BOX + f'player_box_{year}.parquet', f'espn_player_box_{year}.parquet')
    cols = ['game_id', 'athlete_id', 'athlete_display_name', 'athlete_position_abbreviation', 'team_id',
            'team_location', 'team_display_name', 'team_short_display_name', 'minutes', 'field_goals_made',
            'field_goals_attempted', 'three_point_field_goals_made', 'three_point_field_goals_attempted',
            'free_throws_made', 'free_throws_attempted', 'offensive_rebounds', 'defensive_rebounds',
            'assists', 'steals', 'blocks', 'turnovers', 'fouls', 'points', 'team_score', 'opponent_team_score',
            'home_away', 'opponent_team_id']
    box = pd.read_parquet(path, columns=cols)
    box = box.rename(columns={'field_goals_made': 'fgm', 'field_goals_attempted': 'fga',
                              'three_point_field_goals_made': 'fg3m', 'three_point_field_goals_attempted': 'fg3a',
                              'free_throws_made': 'ftm', 'free_throws_attempted': 'fta',
                              'offensive_rebounds': 'orb', 'defensive_rebounds': 'drb', 'assists': 'ast',
                              'steals': 'stl', 'blocks': 'blk', 'turnovers': 'tov', 'fouls': 'pf',
                              'points': 'pts', 'minutes': 'min'})
    box = box[box['min'] > 0].copy()
    stats = COUNTS
    for c in stats:
        box[c] = box[c].astype(float)

    # team and opponent totals for every game
    team = box.groupby(['game_id', 'team_id'], as_index=False)[stats].sum()
    opp = team.rename(columns={'team_id': 'opp_id'})
    team = team.merge(opp, on='game_id', suffixes=('', '_opp'))
    team = team[team['team_id'] != team['opp_id']]
    team = team.rename(columns={c: 'tm_' + c for c in stats}).rename(columns={c + '_opp': 'op_' + c for c in stats})
    box = box.merge(team.drop(columns='opp_id'), on=['game_id', 'team_id'], how='left')

    # how good was his team: average margin, adjusted once for the schedule
    games = box.drop_duplicates(['game_id', 'team_id'])[['game_id', 'team_id', 'team_score', 'opponent_team_score']]
    games['margin'] = games['team_score'] - games['opponent_team_score']
    games = games.merge(games[['game_id', 'team_id']].rename(columns={'team_id': 'opp_id'}), on='game_id')
    games = games[games['team_id'] != games['opp_id']]
    team_margin = games.groupby('team_id')['margin'].mean()
    games['opp_margin'] = games['opp_id'].map(team_margin)
    srs = games.groupby('team_id').agg(team_margin=('margin', 'mean'), opp_margin=('opp_margin', 'mean'))
    srs['team_srs'] = srs['team_margin'] + srs['opp_margin']

    # and the proper version: ridge ratings, and the average opponent each player actually faced
    ratings = team_ratings(box)
    box['opp_rating'] = box['opponent_team_id'].map(ratings)
    box['min_x_opp'] = box['min'] * box['opp_rating']
    slopes = schedule_slopes(box)

    tot_cols = stats + ['tm_' + c for c in stats] + ['op_' + c for c in stats]
    season = box.groupby('athlete_id')[tot_cols + ['min_x_opp']].sum()
    season['sos'] = season.pop('min_x_opp') / season['min']
    season['games'] = box.groupby('athlete_id').size()
    first = box.groupby(['athlete_id', 'team_id']).size().reset_index(name='n').sort_values('n')
    first = first.drop_duplicates('athlete_id', keep='last').set_index('athlete_id')['team_id']
    season['team_id'] = first
    season['team_rating'] = season['team_id'].map(ratings)
    names = box.drop_duplicates('athlete_id').set_index('athlete_id')
    for c in ['athlete_display_name', 'athlete_position_abbreviation', 'team_location', 'team_display_name',
              'team_short_display_name']:
        season[c] = names[c]
    season = season.join(srs[['team_margin', 'team_srs']], on='team_id')
    season['college_season'] = year
    return season.reset_index(), slopes


def rates(s):
    # per 40, shooting and rate stats from season totals
    f = pd.DataFrame(index=s.index)
    m = s['min']
    tm_min = s['tm_min'] / 5
    f['games'] = s['games']
    f['mpg'] = m / s['games']
    for c in PER40:
        f[c + '40'] = 40 * s[c] / m
    f['trb40'] = f['orb40'] + f['drb40']
    f['ts'] = s['pts'] / (2 * (s['fga'] + 0.44 * s['fta']))
    f['efg'] = (s['fgm'] + 0.5 * s['fg3m']) / s['fga']
    f['fg3_pct'] = s['fg3m'] / s['fg3a']
    f['ft_pct'] = s['ftm'] / s['fta']
    f['fg3a_rate'] = s['fg3a'] / s['fga']
    f['fta_rate'] = s['fta'] / s['fga']
    f['usg'] = 100 * (s['fga'] + 0.44 * s['fta'] + s['tov']) * tm_min / (m * (s['tm_fga'] + 0.44 * s['tm_fta'] + s['tm_tov']))
    f['ast_pct'] = 100 * s['ast'] / ((m / tm_min) * s['tm_fgm'] - s['fgm'])
    f['orb_pct'] = 100 * s['orb'] * tm_min / (m * (s['tm_orb'] + s['op_drb']))
    f['drb_pct'] = 100 * s['drb'] * tm_min / (m * (s['tm_drb'] + s['op_orb']))
    opp_poss = s['op_fga'] + 0.44 * s['op_fta'] - s['op_orb'] + s['op_tov']
    f['stl_pct'] = 100 * s['stl'] * tm_min / (m * opp_poss)
    f['blk_pct'] = 100 * s['blk'] * tm_min / (m * (s['op_fga'] - s['op_fg3a']))
    f['tov_pct'] = 100 * s['tov'] / (s['fga'] + 0.44 * s['fta'] + s['tov'])
    f['ast_tov'] = s['ast'] / s['tov'].replace(0, np.nan)
    return f.replace([np.inf, -np.inf], np.nan)


def college_features(s):
    f = pd.DataFrame({'espn_id': s['athlete_id'], 'college_season': s['college_season'],
                      'espn_name': s['athlete_display_name'], 'espn_team': s['team_display_name'],
                      'espn_pos': s['athlete_position_abbreviation']})
    f = pd.concat([f, rates(s)], axis=1)
    for c in ['team_margin', 'team_srs', 'team_rating', 'sos']:
        f[c] = s[c]
    f['name_key'] = s['athlete_display_name'].apply(clean_name)
    f['school_keys'] = [[clean_school(a), clean_school(b), clean_school(c)] for a, b, c in
                        zip(s['team_location'], s['team_display_name'], s['team_short_display_name'])]
    return f


def pos_group(p):
    p = str(p)
    return p[0] if p[:1] in ['G', 'F', 'C'] else 'all'


# the stats every prior and adjustment is worked out for, and what each one's sample size is measured in
RATE_STATS = ['pts40', 'orb40', 'drb40', 'trb40', 'ast40', 'stl40', 'blk40', 'tov40', 'pf40', 'fga40', 'fg3a40',
              'fta40', 'ts', 'efg', 'fg3_pct', 'ft_pct', 'fg3a_rate', 'fta_rate', 'usg', 'ast_pct', 'orb_pct',
              'drb_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'ast_tov']
EXPOSURE = {'ts': 'tsa', 'efg': 'fga', 'fg3_pct': 'fg3a', 'ft_pct': 'fta', 'fg3a_rate': 'fga', 'fta_rate': 'fga'}
# rate stats borrow the shrinkage of the per 40 stat they are built from
BORROW = {'usg': 'fga40', 'ast_pct': 'ast40', 'orb_pct': 'orb40', 'drb_pct': 'drb40', 'stl_pct': 'stl40',
          'blk_pct': 'blk40', 'tov_pct': 'tov40', 'ast_tov': 'ast40'}


def college_priors(totals):
    # every d1 player with 300+ minutes: position and season averages, how noisy each stat is,
    # and how much a player's numbers move from one season to the next
    frames = []
    for year, t in totals.items():
        f = rates(t)
        f['tsa'] = t['fga'] + 0.44 * t['fta']
        for c in ['min', 'fga', 'fg3a', 'fta']:
            f[c] = t[c]
        f['athlete_id'] = t['athlete_id']
        f['college_season'] = year
        f['pos'] = t['athlete_position_abbreviation'].apply(pos_group)
        frames.append(f)
    d1 = pd.concat(frames, ignore_index=True)
    d1 = d1[d1['min'] >= 300]

    rows = []
    for (year, pos), g in d1.groupby(['college_season', 'pos']):
        r = {'college_season': year, 'pos': pos}
        r.update(g[RATE_STATS].mean().to_dict())
        rows.append(r)
    for year, g in d1.groupby('college_season'):
        r = {'college_season': year, 'pos': 'all'}
        r.update(g[RATE_STATS].mean().to_dict())
        rows.append(r)
    priors = pd.DataFrame(rows).drop_duplicates(['college_season', 'pos'], keep='last')

    # shrinkage strength by method of moments: noise variance over the real spread between players
    k = {}
    keyed = d1.merge(priors, on=['college_season', 'pos'], suffixes=('', '_prior'))
    for c in RATE_STATS:
        if c in BORROW:
            continue
        n = keyed[EXPOSURE.get(c, 'min')]
        x = keyed[c]
        ok = x.notna() & (n >= (30 if c in EXPOSURE else 300))
        resid = (x - keyed[c + '_prior'])[ok]
        mean = x[ok].mean()
        if c in EXPOSURE:
            noise_unit = mean * (1 - mean) if c not in ['ts', 'fta_rate'] else 0.25
        else:
            noise_unit = 40 * mean
        true_var = resid.var() - (noise_unit / n[ok]).mean()
        k[c] = float(noise_unit / true_var) if true_var > 0 else 1000.0
    for c, b in BORROW.items():
        k[c] = k[b]

    # change from one season to the next for the same player
    nxt = d1.copy()
    nxt['college_season'] -= 1
    pairs = d1.merge(nxt, on=['athlete_id', 'college_season'], suffixes=('', '_next'))
    w = 2 / (1 / pairs['min'] + 1 / pairs['min_next'])
    deltas = {}
    for c in RATE_STATS:
        d = pairs[c + '_next'] - pairs[c]
        ok = d.notna()
        deltas[c] = float(np.average(d[ok], weights=w[ok]))
    print(f'  priors from {len(d1)} d1 player-seasons, {len(pairs)} back to back pairs')
    return priors, k, deltas


def match_history(college, totals):
    # every earlier d1 season for the matched players, by espn's athlete id
    keep = ['person_id', 'college_season', 'athlete_id', 'team_display_name', 'athlete_position_abbreviation',
            'games'] + COUNTS + TEAM_COUNTS + ['sos', 'team_rating', 'team_margin', 'team_srs']
    frames = []
    for year in range(FIRST_COLLEGE_SEASON, LAST_DRAFT + 1):
        t = totals[year]
        c = college[college['college_season'] >= year][['person_id', 'espn_id']]
        hit = t.merge(c, left_on='athlete_id', right_on='espn_id')
        frames.append(hit)
    h = pd.concat(frames)[keep].rename(columns={'athlete_id': 'espn_id', 'team_display_name': 'espn_team',
                                                'athlete_position_abbreviation': 'espn_pos'})
    return h.sort_values(['person_id', 'college_season']).reset_index(drop=True)


def school_score(org, keys):
    org = clean_school(org)
    return max(difflib.SequenceMatcher(None, org, k).ratio() for k in keys)


def find_player(name, org, season):
    key = clean_name(NAME_FIXES.get(name, name))
    hits = season[season['name_key'] == key]
    if len(hits) == 0:
        close = difflib.get_close_matches(key, season['name_key'].unique(), n=5, cutoff=0.8)
        hits = season[season['name_key'].isin(close)]
        if len(hits) == 0:
            return None
        scores = [school_score(org, k) for k in hits['school_keys']]
        best = int(np.argmax(scores))
        return hits.iloc[best] if scores[best] >= 0.6 else None
    if len(hits) == 1:
        return hits.iloc[0]
    scores = [school_score(org, k) for k in hits['school_keys']]
    return hits.iloc[int(np.argmax(scores))]


def match_college(df, seasons):
    rows = []
    for _, p in df[(df['background'] == 'college') & (df['draft_year'] >= FIRST_COLLEGE_SEASON)].iterrows():
        found = None
        for year in [p['draft_year'], p['draft_year'] - 1]:
            if year not in seasons:
                continue
            hit = find_player(p['name'], p['organization'], seasons[year])
            # a player who missed most of his last year is judged on the year before
            if hit is not None and (found is None or found['games'] < 8):
                if found is None or hit['games'] > found['games']:
                    found = hit
            if found is not None and found['games'] >= 8:
                break
        if found is None:
            print(f"  no college match: {p['draft_year']} {p['name']} ({p['organization']})")
            continue
        r = found.drop(['name_key', 'school_keys']).to_dict()
        r['person_id'] = p['person_id']
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    os.makedirs(os.path.join(RAW, 'nba_api'), exist_ok=True)
    os.makedirs(FINAL, exist_ok=True)

    print('draft history..')
    df = get_draft()
    print(f'{len(df)} picks from {FIRST_DRAFT} to {LAST_DRAFT}')
    print(df['background'].value_counts().to_string())

    print('basketball reference tables..')
    adv, picks, info = get_bbref()
    df = link_bbref(df, picks)
    df = df.merge(info, on='player_id', how='left')

    print('combine..')
    df = df.merge(get_combine(), on='person_id', how='left')

    missing = df[df['bbref_birth'].isna()]['person_id'].tolist()
    print(f'nba.com player info for {len(missing)} players without a bbref profile..')
    df = df.merge(get_player_info(missing), on='person_id', how='left')

    birth = pd.to_datetime(df['bbref_birth'], errors='coerce').fillna(pd.to_datetime(df['nba_birth'], errors='coerce'))
    df['birth_date'] = birth.dt.strftime('%Y-%m-%d')
    df['age'] = ((pd.to_datetime(df['draft_year'].astype(str) + '-06-25') - birth).dt.days / 365.25).round(2)
    # listed heights are roughly in shoes, so take an inch off when there's no combine number
    listed_height = df['bbref_height'].fillna(df['nba_height'])
    df['height'] = df['combine_height'].fillna(listed_height - 1)
    df['weight'] = df['combine_weight'].fillna(df['bbref_weight']).fillna(df['nba_weight'])
    df['wing_diff'] = df['wingspan'] - df['combine_height']
    pos = df['combine_pos'].fillna(df['bbref_pos']).fillna(df['nba_pos']).fillna('').astype(str)
    df['position'] = pos.str.replace('Guard', 'G').str.replace('Forward', 'F').str.replace('Center', 'C')
    df['pos_g'] = df['position'].str.contains('G').astype(int)
    df['pos_f'] = df['position'].str.contains('F').astype(int)
    df['pos_c'] = df['position'].str.contains('C').astype(int)

    print('college box scores..')
    seasons, totals, slope_sums = {}, {}, []
    for year in range(FIRST_COLLEGE_SEASON, LAST_DRAFT + 1):
        totals[year], sl = college_season(year)
        slope_sums.append(sl)
        seasons[year] = college_features(totals[year])
        print(f'  {year}: {len(seasons[year])} players')
    college = match_college(df, seasons)
    df = df.merge(college, on='person_id', how='left')

    print('college history, priors and schedule adjustment..')
    history = match_history(college, totals)
    priors, k, deltas = college_priors(totals)
    slopes = {}
    for stat in GAME_STATS:
        num = sum(x[stat][0] for x in slope_sums)
        den = sum(x[stat][1] for x in slope_sums)
        slopes[stat] = num / den
    print('  change per point of opponent rating: ' + ', '.join(f'{c} {v:+.3f}' for c, v in slopes.items()))
    print('  change from one season to the next: ' + ', '.join(f'{c} {v:+.3f}' for c, v in deltas.items()))
    history.round(4).to_csv(os.path.join(FINAL, 'college_seasons.csv'), index=False)
    priors.round(5).to_csv(os.path.join(FINAL, 'college_priors.csv'), index=False)
    with open(os.path.join(FINAL, 'college_adjust.json'), 'w') as f:
        json.dump({'schedule_slope': slopes, 'season_change': deltas, 'shrink_k': k}, f, indent=1)
    df['has_college_stats'] = df['espn_id'].notna().astype(int)
    for bg in ['college', 'international', 'high school', 'g league / ote']:
        df['bg_' + bg.split(' ')[0]] = (df['background'] == bg).astype(int)

    college_rows = df[(df['background'] == 'college') & (df['draft_year'] >= FIRST_COLLEGE_SEASON)]
    rate = college_rows.groupby('draft_year')['has_college_stats'].mean().round(2)
    print('college match rate by class:')
    print(rate.to_string())
    print(f'overall: {college_rows["has_college_stats"].mean():.3f}')

    print('nba outcomes..')
    out = nba_outcomes(df, adv)
    df = add_targets(df, out)

    drop = ['bbref_pos', 'bbref_height', 'bbref_weight', 'bbref_birth', 'nba_birth', 'nba_pos',
            'nba_height', 'nba_weight', 'combine_pos', 'combine_weight']
    df = df.drop(columns=drop)
    df = df.rename(columns={'player_id': 'bbref_id', 'combine_height': 'height_no_shoes'})
    float_cols = df.select_dtypes('float').columns
    df[float_cols] = df[float_cols].round(4)

    df.to_csv(os.path.join(FINAL, 'draft_players.csv'), index=False)
    out.to_csv(os.path.join(FINAL, 'nba_seasons.csv'), index=False)
    print(f"saved {len(df)} players and {len(out)} player-seasons")
    print(f"complete peak outcomes: {df['peak_complete'].sum()}, complete immediate outcomes: {df['imm_complete'].sum()}")
    print('Done!')


if __name__ == '__main__':
    main()
