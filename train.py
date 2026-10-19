import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import shap
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR

from features import make_features

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(ROOT, 'data', 'final')
MODELS = os.path.join(ROOT, 'models')

FIRST_TRAIN = 2005
FIRST_TEST = 2010
INNER_CLASSES = 6      # tuning only ever looks at the last 6 classes before the test class
TRIALS = 40
ALPHAS = [1, 3, 10, 30, 100, 300, 1000]
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
LEVEL = 0.8            # the ranges are 80% intervals

TARGETS = {'peak': 'peak_vorp', 'imm': 'imm_vorp', 'bpm': 'peak_bpm'}
FLOORS = {'peak': 0.0, 'imm': 0.0, 'bpm': -2.0}   # replacement level for each target

# the v3 settings and inputs, kept as the "before" line
V3_PARAMS = dict(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=20,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                 random_state=7, verbose=-1, n_jobs=1)
V3_FEATURES = ['age', 'height', 'weight', 'wingspan', 'wing_diff', 'reach', 'vertical', 'agility', 'sprint',
               'pos_g', 'pos_f', 'pos_c', 'bg_college', 'bg_international', 'bg_high', 'bg_g',
               'games', 'mpg', 'pts40', 'trb40', 'orb40', 'drb40', 'ast40', 'stl40', 'blk40', 'tov40', 'pf40',
               'fga40', 'fg3a40', 'fta40', 'ts', 'efg', 'fg3_pct', 'ft_pct', 'fg3a_rate', 'fta_rate', 'usg',
               'ast_pct', 'orb_pct', 'drb_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'ast_tov', 'team_margin', 'team_srs']


def load_data(options=None):
    df = pd.read_csv(os.path.join(FINAL, 'draft_players.csv'))
    df, feats = make_features(df, options)
    low = df['games'].fillna(0) * df['mpg'].fillna(0) < 100
    df.loc[low, V3_FEATURES[16:]] = np.nan
    df['log_pick'] = np.log(df['pick'])
    return df[df['draft_year'] >= FIRST_TRAIN].reset_index(drop=True), feats


def class_rank(y, years):
    # where each player finished inside his own class, 0 to 1
    return y.groupby(years.values).rank(pct=True)


# ---------- the two models and the blend ----------

def ridge_model(alpha):
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), Ridge(alpha=alpha))


def gbm_model(params, feats):
    p = dict(params)
    age_sign = p.pop('age_monotone', 0)
    mono = [age_sign if f == 'age' else (-1 if f == 'log_pick' else 0) for f in feats]
    return lgb.LGBMRegressor(**p, monotone_constraints=mono, monotone_constraints_method='advanced')


def fit_blend(train, target, feats, params):
    y = class_rank(train[target], train['draft_year'])
    gbm = gbm_model(params['gbm'], feats).fit(train[feats], y)
    ridge = ridge_model(params['alpha']).fit(train[feats], y)
    return {'gbm': gbm, 'ridge': ridge, 'w': params['w'], 'feats': feats}


def predict_parts(fitted, X):
    X = X[fitted['feats']]
    return fitted['gbm'].predict(X), fitted['ridge'].predict(X)


def predict_blend(fitted, X):
    g, r = predict_parts(fitted, X)
    return fitted['w'] * g + (1 - fitted['w']) * r


# ---------- tuning, only ever on classes before the test class ----------

def gbm_params(trial):
    return dict(n_estimators=trial.suggest_int('n_estimators', 100, 800, step=50),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                num_leaves=trial.suggest_int('num_leaves', 3, 15),
                min_child_samples=trial.suggest_int('min_child_samples', 10, 60),
                subsample=trial.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
                reg_lambda=trial.suggest_float('reg_lambda', 0.1, 30, log=True),
                age_monotone=trial.suggest_categorical('age_monotone', [0, -1]),
                random_state=7, verbose=-1, n_jobs=1)


def inner_folds(train, target):
    folds = []
    years = sorted(train['draft_year'].unique())
    # each inner class is predicted from the ones before it, and needs at least two of those
    for vy in years[2:][-INNER_CLASSES:]:
        a = train[train['draft_year'] < vy]
        b = train[train['draft_year'] == vy]
        folds.append((a, class_rank(a[target], a['draft_year']), b, b[target].values))
    return folds


def mean_rho(preds, folds):
    return np.mean([spearmanr(p, f[3])[0] for p, f in zip(preds, folds)])


def tune(train, target, feats, seed):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = inner_folds(train, target)

    def objective(trial):
        p = gbm_params(trial)
        preds = [gbm_model(p, feats).fit(a[feats], ya).predict(b[feats]) for a, ya, b, yb in folds]
        return mean_rho(preds, folds)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(dict({k: V3_PARAMS[k] for k in ['n_estimators', 'learning_rate', 'num_leaves',
                             'min_child_samples', 'subsample', 'colsample_bytree', 'reg_lambda']}, age_monotone=0))
    study.optimize(objective, n_trials=TRIALS)
    gbm = dict(study.best_params, subsample_freq=1, random_state=7, verbose=-1, n_jobs=1)
    params = {'gbm': gbm, 'gbm_inner': study.best_value}
    params.update(tune_ridge_and_weight(folds, feats, gbm))
    return params


def tune_ridge_and_weight(folds, feats, gbm):
    # ridge strength by the same inner classes, then the blend weight from both models' inner predictions
    best = None
    for alpha in ALPHAS:
        preds = [ridge_model(alpha).fit(a[feats], ya).predict(b[feats]) for a, ya, b, yb in folds]
        rho = mean_rho(preds, folds)
        if best is None or rho > best[1]:
            best = (alpha, rho, preds)
    alpha, ridge_rho, ridge_preds = best
    gbm_preds = [gbm_model(gbm, feats).fit(a[feats], ya).predict(b[feats]) for a, ya, b, yb in folds]
    rhos = [mean_rho([w * g + (1 - w) * r for g, r in zip(gbm_preds, ridge_preds)], folds) for w in WEIGHTS]
    w = float(WEIGHTS[int(np.argmax(rhos))])
    return {'alpha': alpha, 'ridge_inner': ridge_rho, 'w': w, 'blend_inner': max(rhos)}


def slot_params(train, target, feats, params):
    # same settings with the draft slot added, only the blend weight is looked at again
    folds = inner_folds(train, target)
    gbm_preds = [gbm_model(params['gbm'], feats).fit(a[feats], ya).predict(b[feats]) for a, ya, b, yb in folds]
    ridge_preds = [ridge_model(params['alpha']).fit(a[feats], ya).predict(b[feats]) for a, ya, b, yb in folds]
    rhos = [mean_rho([w * g + (1 - w) * r for g, r in zip(gbm_preds, ridge_preds)], folds) for w in WEIGHTS]
    return dict(params, w=float(WEIGHTS[int(np.argmax(rhos))]))


# ---------- ranges: conformalised quantile regression ----------

def oof_scores(train, target, feats, params):
    oof = np.zeros(len(train))
    groups = train['draft_year'].values % 5
    for g in range(5):
        fold = groups == g
        f = fit_blend(train[~fold], target, feats, params)
        oof[fold] = predict_blend(f, train[fold])
    return oof


def quantile_curve(score, y, q):
    bins = pd.qcut(pd.Series(score).rank(method='first'), 10, labels=False).values
    t = pd.DataFrame({'score': score, 'y': y, 'bin': bins}).groupby('bin')
    return t['score'].mean().values, np.maximum.accumulate(t['y'].quantile(q).values)


def conformal(score, y, years, floor, level=LEVEL):
    # percentile curves of the outcome against the score, from half the training classes. then the
    # other half, the most recent class included, picks how wide those curves need to be for 80% of
    # its players to land inside. so many careers sit at replacement level that a plain 10th to 90th
    # range covers far more than 80%, and this sorts that out
    cal = (years.max() - years) % 2 == 0
    need = np.ceil((cal.sum() + 1) * level) / cal.sum()
    for t in np.arange(0.3, 1.001, 0.01):
        xs, lo = quantile_curve(score[~cal], y[~cal], (1 - t) / 2)
        _, hi = quantile_curve(score[~cal], y[~cal], (1 + t) / 2)
        inside = (y[cal] >= np.interp(score[cal], xs, lo)) & (y[cal] <= np.interp(score[cal], xs, hi))
        if inside.mean() >= need:
            break
    return {'q_score': xs, 'q_low': np.maximum(lo, floor), 'q_high': hi, 'q_level': float(t), 'floor': floor}


def vorp_map(score, y, years, floor):
    # 20 slices of the score and what past players in each slice averaged, plus the conformal range
    bins = pd.qcut(pd.Series(score).rank(method='first'), 20, labels=False).values
    t = pd.DataFrame({'score': score, 'y': y, 'bin': bins}).groupby('bin')
    m = {'score': t['score'].mean().values, 'vorp': np.maximum.accumulate(t['y'].mean().values)}
    m.update(conformal(score, y, years, floor))
    return m


def ranges(score, m):
    low = np.interp(score, m['q_score'], m['q_low'])
    high = np.interp(score, m['q_score'], m['q_high'])
    return low, high


def to_vorp(score, m):
    xs, ys = np.asarray(m['score']), np.asarray(m['vorp'])
    mid = np.interp(score, xs, ys)
    # keep the last slope going past the top slice so the best prospects don't all tie
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    mid = np.where(score > xs[-1], ys[-1] + slope * (score - xs[-1]), mid)
    low, high = ranges(score, m)
    return mid, np.minimum(low, mid), np.maximum(high, mid)


# ---------- his 2021 approach ----------

class AveragingModels(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)


def fit_v1(X, y):
    imputer = SimpleImputer(strategy='median').fit(X)
    Xi = pd.DataFrame(imputer.transform(X), columns=X.columns)
    rf = RandomForestRegressor(n_estimators=100, random_state=23, n_jobs=1).fit(Xi, y)
    top = list(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:12].index)
    GBoost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.04, max_depth=5, max_features='sqrt',
                                       min_samples_leaf=20, min_samples_split=10, loss='huber', random_state=5)
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=3, learning_rate=0.06, n_estimators=50,
                                  max_bin=55, verbose=-1, n_jobs=1)
    model_svm = make_pipeline(RobustScaler(), SVR(kernel='poly', degree=2, C=1, epsilon=0.1))
    avg = AveragingModels(models=(GBoost, model_lgb, model_svm)).fit(Xi[top], y)
    return imputer, top, avg


def predict_v1(fitted, X):
    imputer, top, avg = fitted
    Xi = pd.DataFrame(imputer.transform(X), columns=X.columns)
    return avg.predict(Xi[top])


# ---------- scoring a class ----------

def top_hits(score, actual, k):
    # how many of the top k by this score were in the actual top k of the class
    top_score = set(np.argsort(-score, kind='stable')[:k])
    top_actual = set(np.argsort(-actual, kind='stable')[:k])
    return len(top_score & top_actual)


def ndcg10(score, actual, floor=0.0):
    # gains are the outcome above replacement, so a bust is worth nothing
    gain = np.maximum(actual - floor, 0)
    return ndcg_score([gain], [score], k=10)


def class_metrics(test, key, model_cols):
    rows = []
    actual = test[TARGETS[key]].values
    for m in model_cols:
        score = test[m].values
        rows.append({'model': m, 'spearman': spearmanr(score, actual)[0],
                     'top5': top_hits(score, actual, 5), 'top10': top_hits(score, actual, 10),
                     'ndcg10': ndcg10(score, actual, FLOORS[key])})
    return rows


# ---------- why a projection is what it is ----------

def blend_shap(fitted, X):
    # tree shap for the gbm, exact linear shap for the ridge, mixed with the blend weight
    X = X[fitted['feats']]
    tree = shap.TreeExplainer(fitted['gbm']).shap_values(X)
    imputer, scaler, ridge = fitted['ridge'].named_steps.values()
    z = scaler.transform(imputer.transform(X))
    lin = ridge.coef_ * z
    n = X.shape[1]
    out = lin[:, :n].copy()
    # the missing value flags count toward the feature they flag
    for i, j in enumerate(imputer.indicator_.features_):
        out[:, j] += lin[:, n + i]
    return fitted['w'] * tree + (1 - fitted['w']) * out


def shap_rows(values, X, test, key):
    rows = []
    for i in range(len(X)):
        for j in np.argsort(-np.abs(values[i]))[:6]:
            rows.append({'person_id': test['person_id'].iloc[i], 'target': key, 'feature': X.columns[j],
                         'value': X.iloc[i, j], 'shap': values[i][j]})
    return rows


# ---------- one outer test class ----------

def one_class(df, feats, key, year):
    target = TARGETS[key]
    done = df[df[target].notna()]
    train = done[done['draft_year'] < year]
    test = done[done['draft_year'] == year].copy()
    y = train[target]

    params = tune(train, target, feats, seed=year)
    fitted = fit_blend(train, target, feats, params)
    test['score'] = predict_blend(fitted, test)
    test['model_gbm'], test['model_ridge'] = predict_parts(fitted, test)
    years = train['draft_year'].values
    m = vorp_map(oof_scores(train, target, feats, params), y.values, years, FLOORS[key])
    test['model'], test['low'], test['high'] = to_vorp(test['score'].values, m)

    pick_feats = feats + ['log_pick']
    pick = slot_params(train, target, pick_feats, params)
    test['model_pick'] = predict_blend(fit_blend(train, target, pick_feats, pick), test)
    # the same kind of range for the model with the slot, and for the pick number on its own
    mp = conformal(oof_scores(train, target, pick_feats, pick), y.values, years, FLOORS[key])
    test['pick_low'], test['pick_high'] = ranges(test['model_pick'].values, mp)
    md = conformal(-train['log_pick'].values, y.values, years, FLOORS[key])
    test['draft_low'], test['draft_high'] = ranges(-test['log_pick'].values, md)

    v3 = lgb.LGBMRegressor(**V3_PARAMS).fit(train[V3_FEATURES], class_rank(y, train['draft_year']))
    test['v3'] = v3.predict(test[V3_FEATURES])
    test['v1_average'] = predict_v1(fit_v1(train[V3_FEATURES], y), test[V3_FEATURES])
    test['draft_order'] = -test['pick']

    metrics = []
    cols = ['score', 'model_gbm', 'model_ridge', 'model_pick', 'v3', 'v1_average', 'draft_order']
    for r in class_metrics(test, key, cols):
        r['model'] = 'model' if r['model'] == 'score' else r['model']
        r.update({'target': key, 'year': year, 'n': len(test)})
        metrics.append(r)
    for i, lo, hi in [(0, 'low', 'high'), (3, 'pick_low', 'pick_high'), (6, 'draft_low', 'draft_high')]:
        metrics[i]['coverage'] = ((test[target] >= test[lo]) & (test[target] <= test[hi])).mean()
        metrics[i]['width'] = (test[hi] - test[lo]).mean()

    drivers = shap_rows(blend_shap(fitted, test), test[feats], test, key)
    test['target'] = key
    test['actual'] = test[target]
    preds = test[['person_id', 'name', 'draft_year', 'pick', 'target', 'actual', 'score', 'model', 'low', 'high',
                  'model_gbm', 'model_ridge', 'model_pick', 'pick_low', 'pick_high', 'draft_low', 'draft_high', 'v3', 'v1_average']]
    tuned = dict(params['gbm'], alpha=params['alpha'], w=params['w'], w_pick=pick['w'],
                 gbm_inner=params['gbm_inner'], ridge_inner=params['ridge_inner'],
                 blend_inner=params['blend_inner'], target=key, year=year)
    r = {x['model']: x['spearman'] for x in metrics}
    print(f"  {key} {year}: model {r['model']:.2f} (gbm {r['model_gbm']:.2f}, ridge {r['model_ridge']:.2f}, w {params['w']})"
          f"  +pick {r['model_pick']:.2f}  v3 {r['v3']:.2f}  draft order {r['draft_order']:.2f}", flush=True)
    return preds, pd.DataFrame(metrics), pd.DataFrame(drivers), tuned


def final_model(df, feats, key):
    target = TARGETS[key]
    done = df[df[target].notna()]
    params = tune(done, target, feats, seed=1)
    fitted = fit_blend(done, target, feats, params)
    fitted['map'] = vorp_map(oof_scores(done, target, feats, params), done[target].values,
                             done['draft_year'].values, FLOORS[key])
    fitted['last_class'] = int(done['draft_year'].max())
    fitted['params'] = params
    joblib.dump(fitted, os.path.join(MODELS, f'{key}.pkl'))

    sv = blend_shap(fitted, done)
    imp = pd.DataFrame({'feature': feats, 'mean_abs_shap': np.abs(sv).mean(axis=0),
                        'gain': fitted['gbm'].booster_.feature_importance('gain')})
    imp['gain'] = imp['gain'] / imp['gain'].sum()
    imp['target'] = key
    print(f"  {key}: trained on {len(done)} players from {done['draft_year'].min()} to {done['draft_year'].max()}, "
          f"blend weight on the gbm {params['w']}")
    return imp.sort_values('mean_abs_shap', ascending=False)


def target_audit(df):
    # how the old and new coding treat players who barely played, and what it does to the draft order baseline
    seasons = pd.read_csv(os.path.join(FINAL, 'nba_seasons.csv'))
    mp2 = seasons[seasons['k'] <= 2].groupby('person_id')['mp'].sum()
    rows = []
    for col, label in [('imm_vorp_v3', 'Immediate, v3 coding'), ('imm_vorp', 'Immediate, floored'),
                       ('peak_vorp_v3', 'Peak, v3 coding'), ('peak_vorp', 'Peak, floored'), ('peak_bpm', 'Peak BPM')]:
        d = df[df[col].notna() & (df['draft_year'] >= FIRST_TEST)].copy()
        d['mp2'] = d['person_id'].map(mp2).fillna(0)
        d['r'] = d.groupby('draft_year')[col].rank(pct=True)
        rho = np.mean([spearmanr(-g['pick'], g[col])[0] for _, g in d.groupby('draft_year')])
        rows.append({'target': label, 'draft_order_spearman': rho, 'players': len(d),
                     'never_played_rank': d[d['mp2'] == 0]['r'].median(),
                     'played_500_rank': d[d['mp2'] >= 500]['r'].median(),
                     'share_below_zero': (d[col] < 0).mean() if 'bpm' not in col else np.nan})
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    return out


def main():
    os.makedirs(MODELS, exist_ok=True)
    df, feats = load_data()
    print(f'{len(df)} players drafted since {FIRST_TRAIN}, {len(feats)} features')
    target_audit(df).round(4).to_csv(os.path.join(FINAL, 'target_audit.csv'), index=False)

    jobs = []
    for key in TARGETS:
        last = int(df[df[TARGETS[key]].notna()]['draft_year'].max())
        jobs += [(key, year) for year in range(FIRST_TEST, last + 1)]
    print(f'nested backtest, {len(jobs)} test classes..')
    out = Parallel(n_jobs=min(len(jobs), 24))(delayed(one_class)(df, feats, key, year) for key, year in jobs)

    print('final models..')
    imps = Parallel(n_jobs=len(TARGETS))(delayed(final_model)(df, feats, key) for key in TARGETS)

    preds = pd.concat([o[0] for o in out])
    metrics = pd.concat([o[1] for o in out])
    preds.round(4).to_csv(os.path.join(FINAL, 'backtest_predictions.csv'), index=False)
    metrics.round(4).to_csv(os.path.join(FINAL, 'backtest_metrics.csv'), index=False)
    pd.concat([o[2] for o in out]).round(4).to_csv(os.path.join(FINAL, 'backtest_drivers.csv'), index=False)
    pd.DataFrame([o[3] for o in out]).round(4).to_csv(os.path.join(FINAL, 'tuned_params.csv'), index=False)
    pd.concat(imps).round(5).to_csv(os.path.join(FINAL, 'feature_importance.csv'), index=False)
    last = {key: int(df[df[TARGETS[key]].notna()]['draft_year'].max()) for key in TARGETS}
    with open(os.path.join(MODELS, 'info.json'), 'w') as f:
        json.dump({'features': feats, 'last_complete_class': last}, f, indent=2)

    print('\naverage over the outer test classes:')
    summary = metrics.groupby(['target', 'model'])[['spearman', 'top10', 'ndcg10', 'coverage', 'width']].mean().round(3)
    print(summary.to_string())
    print('Done!')


if __name__ == '__main__':
    main()
