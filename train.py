import os
import json
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(ROOT, 'data', 'final')
MODELS = os.path.join(ROOT, 'models')

FIRST_TRAIN = 2005
FIRST_TEST = 2010

FEATURES = ['age', 'height', 'weight', 'wingspan', 'wing_diff', 'reach', 'vertical', 'agility', 'sprint',
            'pos_g', 'pos_f', 'pos_c', 'bg_college', 'bg_international', 'bg_high', 'bg_g',
            'games', 'mpg', 'pts40', 'trb40', 'orb40', 'drb40', 'ast40', 'stl40', 'blk40', 'tov40', 'pf40',
            'fga40', 'fg3a40', 'fta40', 'ts', 'efg', 'fg3_pct', 'ft_pct', 'fg3a_rate', 'fta_rate', 'usg',
            'ast_pct', 'orb_pct', 'drb_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'ast_tov', 'team_margin', 'team_srs']

TARGETS = {'peak': 'peak_vorp', 'imm': 'imm_vorp'}

LGB_PARAMS = dict(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                  random_state=7, verbose=-1)

# with the draft slot added, later picks can only ever lower the projection
PICK_PARAMS = dict(LGB_PARAMS, n_estimators=300, num_leaves=5, min_child_samples=30,
                   monotone_constraints=[0] * len(FEATURES) + [-1])


def load_data():
    df = pd.read_csv(os.path.join(FINAL, 'draft_players.csv'))
    # very low minute college seasons are mostly noise, so treat them like no stats at all
    college_cols = FEATURES[16:]
    low = df['games'].fillna(0) * df['mpg'].fillna(0) < 100
    df.loc[low, college_cols] = np.nan
    return df[df['draft_year'] >= FIRST_TRAIN].reset_index(drop=True)


def point_model(params=None):
    return lgb.LGBMRegressor(**(params or LGB_PARAMS))


def class_rank(y, years):
    # where each player finished inside his own class, 0 to 1
    return y.groupby(years.values).rank(pct=True)


def fit_point(X, y, years, params=None):
    # the model learns to rank players inside a class, which is what the draft is
    model = point_model(params).fit(X, class_rank(y, years))
    # then out of fold scores are mapped back to vorp, so the ratings keep their units
    oof = np.zeros(len(X))
    groups = years.values % 5
    for g in range(5):
        fold = groups == g
        m = point_model(params).fit(X[~fold], class_rank(y[~fold], years[~fold]))
        oof[fold] = m.predict(X[fold])
    return model, vorp_map(oof, y)


def vorp_map(score, y):
    # 20 slices of the score: what past players in each slice averaged, and their 10th and 90th percentiles
    bins = pd.qcut(pd.Series(score).rank(method='first'), 20, labels=False).values
    t = pd.DataFrame({'score': score, 'y': y.values, 'bin': bins}).groupby('bin')
    m = {'score': t['score'].mean().values,
         'vorp': t['y'].mean().values,
         'low': t['y'].quantile(0.1).values,
         'high': t['y'].quantile(0.9).values}
    # a better score should never mean a worse outcome
    for k in ['vorp', 'low', 'high']:
        m[k] = np.maximum.accumulate(m[k])
    return m


def to_vorp(score, m):
    xs = np.asarray(m['score'])
    mid, low, high = [np.interp(score, xs, np.asarray(m[k])) for k in ['vorp', 'low', 'high']]
    # keep the last slope going past the top slice so the best prospects don't all tie
    ys = np.asarray(m['vorp'])
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    mid = np.where(score > xs[-1], ys[-1] + slope * (score - xs[-1]), mid)
    return mid, np.minimum(low, mid), np.maximum(high, mid)


def predict_point(fitted, X):
    model, m = fitted
    score = model.predict(X)
    mid, low, high = to_vorp(score, m)
    return score, mid, low, high


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
    rf = RandomForestRegressor(n_estimators=100, random_state=23).fit(Xi, y)
    top = list(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:12].index)
    GBoost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.04, max_depth=5, max_features='sqrt',
                                       min_samples_leaf=20, min_samples_split=10, loss='huber', random_state=5)
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=3, learning_rate=0.06, n_estimators=50,
                                  max_bin=55, verbose=-1)
    model_svm = make_pipeline(RobustScaler(), SVR(kernel='poly', degree=2, C=1, epsilon=0.1))
    avg = AveragingModels(models=(GBoost, model_lgb, model_svm)).fit(Xi[top], y)
    return imputer, top, avg


def predict_v1(fitted, X):
    imputer, top, avg = fitted
    Xi = pd.DataFrame(imputer.transform(X), columns=X.columns)
    return avg.predict(Xi[top])


def top_hits(score, actual, k):
    # how many of the top k by this score were in the actual top k of the class
    top_score = set(np.argsort(-score, kind='stable')[:k])
    top_actual = set(np.argsort(-actual, kind='stable')[:k])
    return len(top_score & top_actual)


def class_metrics(test, target, model_cols):
    rows = []
    actual = test[target].values
    for m in model_cols:
        score = test[m].values
        rows.append({'model': 'model' if m == 'score' else m, 'spearman': spearmanr(score, actual)[0],
                     'top5': top_hits(score, actual, 5), 'top10': top_hits(score, actual, 10)})
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


def backtest(df, key):
    target = TARGETS[key]
    done = df[df[target].notna()]
    last = int(done['draft_year'].max())
    preds, metrics, drivers = [], [], []
    for year in range(FIRST_TEST, last + 1):
        train = done[done['draft_year'] < year]
        test = done[done['draft_year'] == year].copy()
        X, y = train[FEATURES], train[target]
        Xt = test[FEATURES]

        fitted = fit_point(X, y, train['draft_year'])
        model = fitted[0]
        test['score'], test['model'], test['low'], test['high'] = predict_point(fitted, Xt)

        with_pick = point_model(PICK_PARAMS).fit(train[FEATURES + ['pick']], class_rank(y, train['draft_year']))
        test['model_pick'] = with_pick.predict(test[FEATURES + ['pick']])
        test['v1_average'] = predict_v1(fit_v1(X, y), Xt)
        test['draft_order'] = -test['pick']

        for r in class_metrics(test, target, ['score', 'model_pick', 'v1_average', 'draft_order']):
            r.update({'target': key, 'year': year, 'n': len(test)})
            metrics.append(r)
        drivers += shap_rows(model, Xt, test, key)
        test['target'] = key
        test['actual'] = test[target]
        preds.append(test[['person_id', 'name', 'draft_year', 'pick', 'target', 'actual', 'score', 'model', 'low', 'high',
                           'model_pick', 'v1_average']])
        m = metrics[-4:]
        print(f"  {key} {year}: model {m[0]['spearman']:.2f}  +pick {m[1]['spearman']:.2f}  "
              f"v1 {m[2]['spearman']:.2f}  draft order {m[3]['spearman']:.2f}")
    return pd.concat(preds), pd.DataFrame(metrics), pd.DataFrame(drivers)


def final_models(df, key):
    target = TARGETS[key]
    done = df[df[target].notna()]
    X, y = done[FEATURES], done[target]
    model, m = fit_point(X, y, done['draft_year'])
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

    all_preds, all_metrics, all_drivers, all_imp, last_year = [], [], [], [], {}
    for key in TARGETS:
        print(f'backtest for {key}..')
        p, m, d = backtest(df, key)
        all_preds.append(p)
        all_metrics.append(m)
        all_drivers.append(d)

    print('final models..')
    for key in TARGETS:
        imp, last = final_models(df, key)
        all_imp.append(imp)
        last_year[key] = last

    preds = pd.concat(all_preds)
    metrics = pd.concat(all_metrics)
    preds.round(4).to_csv(os.path.join(FINAL, 'backtest_predictions.csv'), index=False)
    metrics.round(4).to_csv(os.path.join(FINAL, 'backtest_metrics.csv'), index=False)
    pd.concat(all_drivers).round(4).to_csv(os.path.join(FINAL, 'backtest_drivers.csv'), index=False)
    pd.concat(all_imp).round(5).to_csv(os.path.join(FINAL, 'feature_importance.csv'), index=False)
    with open(os.path.join(MODELS, 'info.json'), 'w') as f:
        json.dump({'features': FEATURES, 'last_complete_class': last_year}, f, indent=2)

    print('\naverage over backtest classes:')
    summary = metrics.groupby(['target', 'model'])[['spearman', 'top5', 'top10']].mean().round(3)
    print(summary.to_string())
    for key in TARGETS:
        p = preds[preds['target'] == key]
        inside = ((p['actual'] >= p['low']) & (p['actual'] <= p['high'])).mean()
        print(f'{key}: {inside:.1%} of actual outcomes landed inside the 10-90 range')
    print('Done!')


if __name__ == '__main__':
    main()
