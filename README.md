# nba_draft

Projecting how NBA draft prospects will turn out, using their college and pre-draft numbers.

**Site (V4):** https://statxsphere.github.io/nba_draft/
**V2 app:** https://statosphere-nba-draft-projections.streamlit.app/

## What it does

I started this before the 2021 draft, came back to it in 2024, and rebuilt it twice in 2026.

**V1 (2021).** I scraped draft classes from Tankathon and advanced stats from Basketball Reference, matched the players up and trained regression models to predict each prospect's peak VORP in the NBA. I also tried classifying player archetypes, which didn't work well enough to use. The whole process is in `analysis_module/draft projections.ipynb`, and I wrote it up here: [NBA Draft 2021: an analysis](https://shoobydo.substack.com/p/nba-draft-2021-an-analysis).

**V2 (2024).** I changed what the model predicts to two ratings, both scaled from -10 to 10:

- **Peak Impact Rating**: the player's best 3-season rolling VORP.
- **Immediate Impact Rating**: how they do in their first 2 seasons.

For player comps, I clustered past draftees and used cosine similarity to pick a ceiling and a floor comparison for each prospect. The app plots those two players' ratings season by season. V2 covers the 2023 and 2024 classes, and its results are in `data/final/temp.csv`.

**V3 (2026).** Neither version ever checked itself against the obvious baseline, which is the order teams actually drafted in. V3 did that, on a rebuilt dataset: every pick from 2000 to 2026 (1,602 players), combine measurements, college box scores and NBA outcomes through 2025-26, a LightGBM model, SHAP drivers, V2's comps and a static site.

**V4 (2026).** V3 had two problems. Its settings were picked on the same backtest it reported, and the way it counted busts broke the draft-order baseline. V4:

- tunes everything only on classes before the one being tested (nested, time-ordered cross validation with Optuna), so the results below are honest
- counts a season below replacement as replacement, the same as a season not played
- adds two per-possession targets that don't reward a pick for just getting minutes: Peak BPM, and my own plus-minus (RAPM) built from every NBA lineup since 2007-08
- blends a ridge regression with LightGBM, uses the last three college seasons instead of one, and adds shot types from play-by-play
- gives 80% ranges calibrated on earlier classes, and checks how often they held
- has two boards' worth of numbers: stats only (the headline) and stats + draft slot

## V4 results

For every class from 2010 on, the model trains and tunes only on earlier drafts and then ranks the new class. These are averages over those test classes. Spearman is the rank correlation with what the class actually did, top 10 is how many of the real top 10 it had in its own top 10, and NDCG@10 rewards getting the best players at the very top. The draft-order baseline is just the pick number.

| | Stats only | Stats + draft slot | Draft order |
| --- | --- | --- | --- |
| **Peak** (2010-2020, 11 classes) | 0.475 · 3.9 · 0.463 | 0.574 · 4.0 · 0.530 | 0.568 · 4.0 · 0.539 |
| **Immediate** (2010-2024, 15 classes) | 0.414 · 4.2 · 0.482 | 0.479 · 4.5 · 0.532 | 0.416 · 4.1 · 0.533 |
| **Peak BPM** (2010-2020) | 0.447 · 3.3 · 0.497 | 0.526 · 3.5 · 0.549 | 0.509 · 3.8 · 0.546 |
| **Peak RAPM** (2010-2020) | 0.420 · 3.8 · 0.505 | 0.505 · 4.0 · 0.558 | 0.470 · 3.5 · 0.513 |
| **Immediate RAPM** (2010-2024) | 0.322 · 3.5 · 0.455 | 0.331 · 3.5 · 0.450 | 0.261 · 3.1 · 0.398 |
| Classes where stats only beats the draft order | Peak 2/11, Immediate 9/15, BPM 4/11, RAPM 2/11, Imm. RAPM 12/15 | | |
| Real outcome inside the 80% range | Peak 81%, Immediate 84%, BPM 79%, RAPM 83% | Peak 82%, Immediate 84% | |

Each cell is Spearman · top 10 hits · NDCG@10.

For comparison, V3 as published claimed 0.374 against 0.447 on Peak and 0.289 against 0.059 on Immediate. Run on the fixed targets with its old settings, V3 gets 0.449 on Peak and 0.373 on Immediate, so V4 is a real step up from V3, but the old Immediate "win" over the teams was mostly the bust-counting bug.

So on stats alone the teams still win on the long-run questions and it's about even on the first two years. The pre-draft numbers do add something the teams miss on some targets: with the slot added, the model clearly beats the slot on its own on Immediate and both plus-minus targets, and is level with it on Peak and Peak BPM. The slot version reuses the stats-only settings and only picks its blend weight again.

The model's top ten in the 2026 class by projected Peak Impact Rating, stats only:

| Model | Player | Pick | Peak | Immediate |
| --- | --- | --- | --- | --- |
| 1 | Caleb Wilson | 4 | 7.4 | 6.9 |
| 2 | Cameron Boozer | 3 | 7.1 | 8.2 |
| 3 | Brayden Burries | 10 | 5.7 | 4.0 |
| 4 | Kingston Flemings | 8 | 5.5 | 5.0 |
| 5 | AJ Dybantsa | 1 | 5.4 | 4.7 |
| 6 | Keaton Wagler | 5 | 5.1 | 2.6 |
| 7 | Ebuka Okorie | 17 | 5.0 | 3.4 |
| 8 | Hannes Steinbach | 14 | 5.0 | 4.1 |
| 9 | Allen Graves | 19 | 4.9 | 5.3 |
| 10 | Dailyn Swain | 15 | 4.4 | 2.8 |

Their 80% ranges all run from about replacement level to a 10, which is how uncertain a top-20 pick really is.

## How V4 is built

**Targets.** Peak is the best three-season run of VORP within a player's first six seasons, averaged per season, and Immediate is average VORP over the first two. Any season below replacement counts as replacement (0), the same as a season not played. Peak BPM is the best three-season run of BPM in seasons 2 to 6, minutes pooled with 750 minutes of replacement level (−2) added. Peak RAPM is the best three-season window of my own plus-minus in seasons 2 to 6, and Immediate RAPM is the first two seasons. 4 VORP a season (Peak) or 2 (Immediate) is a 10. Peak targets are graded up to the 2020 class and Immediate up to 2024.

**RAPM.** `rapm.py` pulls every five-man lineup's possessions and offensive and defensive rating from stats.nba.com (2007-08 to 2025-26), then fits a ridge regression on who was on the floor over three-season windows, weighted by possessions, with everyone starting at replacement level. The ridge strength came from cross validation on the lineups. It can't see opponents, so it's rougher than possession-level RAPM, but it correlates 0.78 with FiveThirtyEight's RAPTOR over 2013-14 to 2021-22 (0.81 on offence, 0.61 on defence, 633 players).

**Model.** A blend of LightGBM and ridge regression, both predicting where a player finishes inside his own class. For each test class, Optuna tunes LightGBM on the six classes before it (each predicted from the ones before that), and the ridge strength and blend weight are picked the same way. Scores go back to VORP (or BPM, or plus-minus) through 20 slices of out-of-fold scores. The draft slot is left out of the headline model; "stats + draft slot" adds log(pick) with the constraint that a later pick can only lower the projection.

**What didn't make it.** Opponent-adjusted stats, age-adjusted stats, shrinkage toward position averages, position-relative stats, combine flags, Pelton-style season weights, a ridge per position, a lambdarank head and a two-part "reaches the league × impact" model were all tested and none helped. The code is still there, switched off, and `python ablation.py` reruns the feature comparison. That comparison is the one step that looks at the test classes, so I only switched whole groups on or off with a rule I set before running it.

**Ranges.** 80% ranges, conformalised on earlier classes: percentile curves fit on half of them, and the other half (including the most recent class) picks how wide they need to be.

**Players without college stats.** International, high school and G League / OTE picks stay in with a background flag and empty college features.

The full write-up is on the [method page](https://statxsphere.github.io/nba_draft/method.html).

## Rebuilding it

From the repo root, on Python 3.12:

```
pip install -r requirements-model.txt
python build_dataset.py && python rapm.py && python train.py && python predict.py
```

- `build_dataset.py` downloads and caches everything into `data/raw/` (not in git), then writes `data/final/draft_players.csv`, `nba_seasons.csv` and the college season, prior and adjustment tables.
- `rapm.py` fetches the NBA lineups, fits the plus-minus and adds `peak_rapm` and `imm_rapm` to `draft_players.csv`. Run it after `build_dataset.py`.
- `features.py` turns the college seasons into model inputs. Each idea is one function in a list, switched on or off in `OPTIONS`.
- `train.py` runs the nested backtest (about six minutes on 32 threads), writes `data/final/backtest_*.csv`, `tuned_params.csv` and `feature_importance.csv`, and saves the final models to `models/` (not in git). A new target is one more entry in `TARGETS`.
- `ablation.py` compares the feature ideas and writes `data/final/ablation.csv`.
- `predict.py` writes `data/final/projections.csv` and the JSON in `docs/data/` that the site reads.

To look at the site locally:

```
cd docs
python -m http.server 8622
```

It's plain HTML, CSS and JavaScript with no build step. Charts use Observable Plot from a pinned CDN link. GitHub Pages can serve it straight from `docs/` on `main`.

## Data sources

- [nba_api](https://github.com/swar/nba_api) (stats.nba.com): draft history, draft combine, player info and five-man lineup stats. Calls are spaced out, failed ones are retried slowly, and every response is cached to disk, so a rerun never asks twice.
- [hoopR / SportsDataverse](https://github.com/sportsdataverse/hoopR-mbb-data): ESPN men's college basketball box scores from 2004-05 on and play-by-play for shot types from 2013-14 on, released under CC BY 4.0. They're downloaded once as release files from GitHub.
- [sumitrodatta/bball-reference-datasets](https://github.com/sumitrodatta/bball-reference-datasets): a public CSV mirror of [Basketball Reference](https://www.basketball-reference.com/) tables (season advanced stats, draft picks, player info). The repo has no licence file and the numbers belong to Sports Reference, so I only commit my own derived tables. The snapshot runs through the 2025-26 season.
- [FiveThirtyEight's RAPTOR data](https://github.com/fivethirtyeight/data/tree/master/nba-raptor) (CC BY 4.0), only to check my plus-minus.

There's no web scraping. Sports Reference and Barttorvik both sit behind bot protection now, so instead of scraping them I used the open sources above. CollegeBasketballData would add recruiting ranks and more college splits, but it needs an API key; `build_dataset.py` reads one from `CBBD_API_KEY` if it's ever set.

The committed tables in `data/final/` come to about 4 MB, and the site data in `docs/data/` to about 3.4 MB.

## Repo layout

```
build_dataset.py           downloads and builds the dataset
rapm.py                    my own NBA plus-minus from lineup data, added as targets
features.py                college seasons into model inputs
train.py                   nested backtest and final models
ablation.py                which feature ideas earn their place
predict.py                 projections, comps, SHAP drivers and the site's JSON
docs/                      the site (GitHub Pages)
data/final/                derived tables, plus temp.csv for the V2 app
nba-draft-dashboard.py     the V2 Streamlit app
analysis_module/           V1 notebook (regression + archetype classification)
data_collection_module/    V1 scrapers for Tankathon, Basketball Reference and nba.com
requirements.txt           what the Streamlit app needs
requirements-model.txt     what the V4 scripts need
requirements-dev.txt       extra packages for the V1 scrapers and the notebook
.devcontainer/             Codespaces setup that starts the app
```

## Running the V2 app

From the repo root:

```
pip install -r requirements.txt
streamlit run nba-draft-dashboard.py
```

It opens on http://localhost:8501. You can sort by pick or either rating, filter by draft year and search for a player.

A few notes on the older code:

- The notebook expects the scraped CSVs in `data/`, and those aren't in the repo. You'd have to run the scrapers again to re-run it, and the sites have changed since 2021.
- The nba.com scraper was never finished.
- Run the scrapers and the notebook from the repo root too, since they use relative paths.

More of my projects: https://statxsphere.github.io/#projects
