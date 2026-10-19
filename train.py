import os
import json
import warnings

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
from sklearn.metrics import ndcg_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(ROOT, 'data', 'final')
MODELS = os.path.join(ROOT, 'models')

FIRST_TRAIN = 2005
FIRST_TEST = 2010
INNER_CLASSES = 3      # tuning only ever looks at the last 3 classes before the test class
TRIALS = 40

FEATURES = ['age', 'height', 'weight', 'wingspan', 'wing_diff', 'reach', 'vertical', 'agility', 'sprint',
            'pos_g', 'pos_f', 'pos_c', 'bg_college', 'bg_international', 'bg_high', 'bg_g',
            'games', 'mpg', 'pts40', 'trb40', 'orb40', 'drb40', 'ast40', 'stl40', 'blk40', 'tov40', 'pf40',
            'fga40', 'fg3a40', 'fta40', 'ts', 'efg', 'fg3_pct', 'ft_pct', 'fg3a_rate', 'fta_rate', 'usg',
            'ast_pct', 'orb_pct', 'drb_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'ast_tov', 'team_margin', 'team_srs']

TARGETS = {'peak': 'peak_vorp', 'imm': 'imm_vorp'}

# the v3 settings, kept as the "before" line
V3_PARAMS = dict(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=20,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                 random_state=7, verbose=-1, n_jobs=1)


def load_data():
    df = pd.read_csv(os.path.join(FINAL, 'draft_players.csv'))
    # very low minute college seasons are mostly noise, so treat them like no stats at all
    college_cols = FEATURES[16:]
    low = df['games'].fillna(0) * df['mpg'].fillna(0) < 100
    df.loc[low, college_cols] = np.nan
    return df[df['draft_year'] >= FIRST_TRAIN].reset_index(drop=True)


def class_rank(y, years):
    # where each player finished inside his own class, 0 to 1
    return y.groupby(years.values).rank(pct=True)


def class_spearman(score, actual, years):
    out = []
    for y in np.unique(years):
        m = years == y
        out.append(spearmanr(score[m], actual[m])[0])
    return np.mean(out)


# ---------- tuning, only ever on classes before the test class ----------

def gbm_params(trial):
    return dict(n_estimators=trial.suggest_int('n_estimators', 100, 800, step=50),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                num_leaves=trial.suggest_int('num_leaves', 3, 15),
                min_child_samples=trial.suggest_int('min_child_samples', 10, 60),
                subsample=trial.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
                reg_lambda=trial.suggest_float('reg_lambda', 0.1, 30, log=True),
                random_state=7, verbose=-1, n_jobs=1)


def inner_years(train):
    years = sorted(train['draft_year'].unique())
    return years[-INNER_CLASSES:]


def tune(train, target, feats, seed):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = []
    for vy in inner_years(train):
        a = train[train['draft_year'] < vy]
        b = train[train['draft_year'] == vy]
        folds.append((a[feats], class_rank(a[target], a['draft_year']), b[feats], b[target].values))

    def objective(trial):
        p = gbm_params(trial)
        rhos = []
        for Xa, ya, Xb, yb in folds:
            m = lgb.LGBMRegressor(**p).fit(Xa, ya)
            rhos.append(spearmanr(m.predict(Xb), yb)[0])
        return np.mean(rhos)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({k: v for k, v in V3_PARAMS.items() if k in ['n_estimators', 'learning_rate', 'num_leaves',
                         'min_child_samples', 'subsample', 'colsample_bytree', 'reg_lambda']})
    study.optimize(objective, n_trials=TRIALS)
    p = dict(study.best_params, subsample_freq=1, random_state=7, verbose=-1, n_jobs=1)
    return p, study.best_value


# ---------- the old ranges, out of fold score slices ----------

def fit_point(X, y, years, params):
    model = lgb.LGBMRegressor(**params).fit(X, class_rank(y, years))
    oof = np.zeros(len(X))
    groups = years.values % 5
    for g in range(5):
        fold = groups == g
        m = lgb.LGBMRegressor(**params).fit(X[~fold], class_rank(y[~fold], years[~fold]))
        oof[fold] = m.predict(X[fold])
    return model, vorp_map(oof, y)


def vorp_map(score, y):
    bins = pd.qcut(pd.Series(score).rank(method='first'), 20, labels=False).values
    t = pd.DataFrame({'score': score, 'y': y.values, 'bin': bins}).groupby('bin')
    m = {'score': t['score'].mean().values,
         'vorp': t['y'].mean().values,
         'low': t['y'].quantile(0.1).values,
         'high': t['y'].quantile(0.9).values}
    for k in ['vorp', 'low', 'high']:
        m[k] = np.maximum.accumulate(m[k])
    return m


def to_vorp(score, m):
    xs = np.asarray(m['score'])
    mid, low, high = [np.interp(score, xs, np.asarray(m[k])) for k in ['vorp', 'low', 'high']]
    ys = np.asarray(m['vorp'])
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    mid = np.where(score > xs[-1], ys[-1] + slope * (score - xs[-1]), mid)
    return mid, np.minimum(low, mid), np.maximum(high, mid)


# his 2021 approach: impute, keep the 12 features a random forest likes most, then average three models
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
    # gains are the outcome above the floor, so a bust is worth nothing
    gain = np.maximum(actual - floor, 0)
    return ndcg_score([gain], [score], k=10)


def class_metrics(test, target, model_cols):
    rows = []
    actual = test[target].values
    for m in model_cols:
        score = test[m].values
        rows.append({'model': 'model' if m == 'score' else m, 'spearman': spearmanr(score, actual)[0],
                     'top5': top_hits(score, actual, 5), 'top10': top_hits(score, actual, 10),
                     'ndcg10': ndcg10(score, actual)})
    return rows


def shap_rows(model, X, test, key):
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    rows = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(values[i]))[:6]
        for j in order:
            rows.append({'person_id': test['person_id'].iloc[i], 'target': key, 'feature': X.columns[j],
                         'value': X.iloc[i, j], 'shap': values[i][j], 'base': float(explainer.expected_value)})
    return rows


def one_class(df, key, year):
    target = TARGETS[key]
    done = df[df[target].notna()]
    train = done[done['draft_year'] < year]
    test = done[done['draft_year'] == year].copy()
    X, y = train[FEATURES], train[target]
    Xt = test[FEATURES]

    params, inner = tune(train, target, FEATURES, seed=year)
    fitted = fit_point(X, y, train['draft_year'], params)
    model = fitted[0]
    test['score'] = model.predict(Xt)
    test['model'], test['low'], test['high'] = to_vorp(test['score'].values, fitted[1])

    v3 = lgb.LGBMRegressor(**V3_PARAMS).fit(X, class_rank(y, train['draft_year']))
    test['v3'] = v3.predict(Xt)
    pick_params = dict(params, monotone_constraints=[0] * len(FEATURES) + [-1])
    with_pick = lgb.LGBMRegressor(**pick_params).fit(train[FEATURES + ['pick']], class_rank(y, train['draft_year']))
    test['model_pick'] = with_pick.predict(test[FEATURES + ['pick']])
    test['v1_average'] = predict_v1(fit_v1(X, y), Xt)
    test['draft_order'] = -test['pick']

    metrics = []
    for r in class_metrics(test, target, ['score', 'model_pick', 'v3', 'v1_average', 'draft_order']):
        r.update({'target': key, 'year': year, 'n': len(test), 'inner_spearman': inner})
        metrics.append(r)
    inside = (test[target] >= test['low']) & (test[target] <= test['high'])
    metrics[0]['coverage'] = inside.mean()
    drivers = shap_rows(model, Xt, test, key)
    test['target'] = key
    test['actual'] = test[target]
    preds = test[['person_id', 'name', 'draft_year', 'pick', 'target', 'actual', 'score', 'model', 'low', 'high',
                  'model_pick', 'v3', 'v1_average']]
    print(f"  {key} {year}: model {metrics[0]['spearman']:.2f}  +pick {metrics[1]['spearman']:.2f}  "
          f"v3 {metrics[2]['spearman']:.2f}  draft order {metrics[4]['spearman']:.2f}  (inner {inner:.2f})", flush=True)
    return preds, pd.DataFrame(metrics), pd.DataFrame(drivers), dict(params, target=key, year=year)


def final_models(df, key):
    target = TARGETS[key]
    done = df[df[target].notna()]
    X, y = done[FEATURES], done[target]
    params, _ = tune(done, target, FEATURES, seed=1)
    model, m = fit_point(X, y, done['draft_year'], params)
    model.booster_.save_model(os.path.join(MODELS, f'{key}_model.txt'))
    with open(os.path.join(MODELS, f'{key}_vorp_map.json'), 'w') as f:
        json.dump({k: v.tolist() for k, v in m.items()}, f)
    imp = pd.DataFrame({'feature': FEATURES, 'gain': model.booster_.feature_importance('gain')})
    imp['gain'] = imp['gain'] / imp['gain'].sum()
    sv = shap.TreeExplainer(model).shap_values(X)
    imp['mean_abs_shap'] = np.abs(sv).mean(axis=0)
    imp['target'] = key
    print(f'  {key}: trained on {len(done)} players from {done["draft_year"].min()} to {done["draft_year"].max()}')
    return imp.sort_values('mean_abs_shap', ascending=False), int(done['draft_year'].max())


def main():
    os.makedirs(MODELS, exist_ok=True)
    df = load_data()
    print(f'{len(df)} players drafted since {FIRST_TRAIN}')

    jobs = []
    for key in TARGETS:
        last = int(df[df[TARGETS[key]].notna()]['draft_year'].max())
        jobs += [(key, year) for year in range(FIRST_TEST, last + 1)]
    print(f'nested backtest, {len(jobs)} test classes..')
    out = Parallel(n_jobs=min(len(jobs), 24))(delayed(one_class)(df, key, year) for key, year in jobs)

    print('final models..')
    all_imp, last_year = [], {}
    for key in TARGETS:
        imp, last = final_models(df, key)
        all_imp.append(imp)
        last_year[key] = last

    preds = pd.concat([o[0] for o in out])
    metrics = pd.concat([o[1] for o in out])
    preds.round(4).to_csv(os.path.join(FINAL, 'backtest_predictions.csv'), index=False)
    metrics.round(4).to_csv(os.path.join(FINAL, 'backtest_metrics.csv'), index=False)
    pd.concat([o[2] for o in out]).round(4).to_csv(os.path.join(FINAL, 'backtest_drivers.csv'), index=False)
    pd.DataFrame([o[3] for o in out]).to_csv(os.path.join(FINAL, 'tuned_params.csv'), index=False)
    pd.concat(all_imp).round(5).to_csv(os.path.join(FINAL, 'feature_importance.csv'), index=False)
    with open(os.path.join(MODELS, 'info.json'), 'w') as f:
        json.dump({'features': FEATURES, 'last_complete_class': last_year}, f, indent=2)

    print('\naverage over the outer test classes:')
    summary = metrics.groupby(['target', 'model'])[['spearman', 'top10', 'ndcg10', 'coverage']].mean().round(3)
    print(summary.to_string())
    print('Done!')


if __name__ == '__main__':
    main()
