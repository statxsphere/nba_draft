# nba_draft

Projecting how NBA draft prospects will turn out, using their college and pre-draft numbers.

**V3 site:** https://statxsphere.github.io/nba_draft/
**V2 app:** https://statosphere-nba-draft-projections.streamlit.app/

## What it does

I started this before the 2021 draft, came back to it in 2024, and rebuilt it in 2026.

**V1 (2021).** I scraped draft classes from Tankathon and advanced stats from Basketball Reference, matched the players up and trained regression models to predict each prospect's peak VORP in the NBA. I also tried classifying player archetypes, which didn't work well enough to use. The whole process is in `analysis_module/draft projections.ipynb`, and I wrote it up here: [NBA Draft 2021: an analysis](https://shoobydo.substack.com/p/nba-draft-2021-an-analysis).

**V2 (2024).** I changed what the model predicts to two ratings, both scaled from -10 to 10:

- **Peak Impact Rating**: the player's best 3-season rolling VORP.
- **Immediate Impact Rating**: how they do in their first 2 seasons.

For player comps, I clustered past draftees and used cosine similarity to pick a ceiling and a floor comparison for each prospect. The app plots those two players' ratings season by season. V2 covers the 2023 and 2024 classes, and its results are in `data/final/temp.csv`.

**V3 (2026).** Neither version ever checked itself against the obvious baseline, which is the order teams actually drafted in. V3 does that, on a rebuilt dataset and model:

- every pick from 2000 to 2026 (1,602 players), with combine measurements, final college season stats and NBA outcomes through 2025-26
- the same two ratings, recomputed for everyone whose career is long enough to grade
- a LightGBM model that ranks each class, backtested one class at a time against the real draft order
- 10th to 90th percentile ranges, SHAP drivers for every player, and V2's cosine-similarity comps, each showing what the comp's career actually turned into
- a static site with a big board for every class from 2010 to 2026, a backtest page and a method page

## V3 results

For every class from 2010 on, the model trains only on earlier drafts and then ranks the new class. The score is the Spearman rank correlation between that ranking and what the class actually did, averaged over the classes. The draft-order baseline is just the pick number.

| | Peak (2010-2020, 11 classes) | Immediate (2010-2024, 15 classes) |
| --- | --- | --- |
| Model, pre-draft stats only | **0.374** | **0.289** |
| Actual draft order | 0.447 | 0.059 |
| Model + draft slot | 0.437 | 0.299 |
| My 2021 averaged regression, same data | 0.327 | 0.193 |
| Classes where the model beats the draft order | 4 of 11 | 13 of 15 |
| Top 10 hits per class, model vs picks 1-10 | 3.5 vs 4.0 | 3.3 vs 3.9 |
| Real outcome inside the 10-90% range | 83% | 74% |

So the teams still win on Peak, which makes sense since they know a lot more than the box score does. Within the first round alone it's close (0.233 vs 0.247). On Immediate the model wins clearly, mostly because the draft order barely predicts early VORP at all: high picks often play big minutes on bad teams before they're ready. On both targets it's a clear step up from my 2021 approach.

The model's top five in the 2026 class by projected Peak Impact Rating:

| Model | Player | Pick | Peak | Immediate |
| --- | --- | --- | --- | --- |
| 1 | Cameron Boozer | 3 | 8.9 | 6.4 |
| 2 | Caleb Wilson | 4 | 8.1 | 6.0 |
| 3 | Kingston Flemings | 8 | 6.6 | 3.5 |
| 4 | Hannes Steinbach | 14 | 5.7 | 3.2 |
| 5 | Brayden Burries | 10 | 5.5 | 1.4 |

AJ Dybantsa (pick 1) comes in 14th and Darryn Peterson (pick 2) 30th. That's what the stats alone say, and the Peak backtest above is a good reminder of how much the teams know that they don't.

## How V3 is built

**Targets.** Peak is the best three-season run of VORP within a player's first six seasons, averaged per season. Immediate is average VORP over the first two seasons. A season he didn't play counts as zero, which is replacement level, so stashed picks and busts stay in. 4 VORP a season (Peak) or 2 (Immediate) is a 10. Careers are still going for recent classes, so Peak is only graded up to the 2020 class and Immediate up to 2024. Newer players are never trained on, and the board shows their numbers so far.

**Model.** LightGBM with small, regularised trees. It learns where each player finished inside his own class, and out-of-fold scores are mapped back to VORP in 20 slices, which also gives the 10th to 90th percentile ranges. The draft slot is left out on purpose; the "model + draft slot" line above is a separate check.

**Players without college stats.** 303 international picks, 36 from high school and 16 from the G League or Overtime Elite aren't dropped. They get a background flag and empty college features, which LightGBM handles natively, so the model judges them on age, measurements and path. Their comps come from other non-college picks. Drafts before 2005 have no college box scores, so those classes are only used for outcomes and comps, not for training.

**Comps.** Pre-draft features are standardised and compared by cosine similarity on whatever both players have, against earlier classes with complete careers. Out of the ten closest, the best career is the ceiling comp and the worst is the floor.

The full write-up is on the [method page](https://statxsphere.github.io/nba_draft/method.html).

## Rebuilding it

From the repo root, on Python 3.12:

```
pip install -r requirements-model.txt
python build_dataset.py && python train.py && python predict.py
```

- `build_dataset.py` downloads and caches everything into `data/raw/` (not in git), then writes `data/final/draft_players.csv` and `data/final/nba_seasons.csv`.
- `train.py` runs the backtest, writes `data/final/backtest_*.csv` and `feature_importance.csv`, and saves the final models to `models/` (not in git).
- `predict.py` writes `data/final/projections.csv` and the JSON in `docs/data/` that the site reads.

To look at the site locally:

```
cd docs
python -m http.server 8622
```

It's plain HTML, CSS and JavaScript with no build step. Charts use Observable Plot from a pinned CDN link. GitHub Pages can serve it straight from `docs/` on `main`.

## Data sources

- [nba_api](https://github.com/swar/nba_api) (stats.nba.com): draft history, draft combine and player info. Calls are spaced out, failed ones are retried slowly, and every response is cached to disk, so a rerun never asks twice.
- [hoopR / SportsDataverse](https://github.com/sportsdataverse/hoopR-mbb-data): ESPN men's college basketball box scores from 2004-05 on, released under CC BY 4.0. They're downloaded once as release files from GitHub.
- [sumitrodatta/bball-reference-datasets](https://github.com/sumitrodatta/bball-reference-datasets): a public CSV mirror of [Basketball Reference](https://www.basketball-reference.com/) tables (season advanced stats, draft picks, player info). The repo has no licence file and the numbers belong to Sports Reference, so I only commit my own derived tables. The snapshot runs through the 2025-26 season.

There's no web scraping in V3. Sports Reference and Barttorvik both sit behind bot protection now, so instead of scraping them I used the open sources above.

The committed tables in `data/final/` come to about 1.7 MB, and the site data in `docs/data/` to about 3 MB.

## Repo layout

```
build_dataset.py           V3: downloads and builds the dataset
train.py                   V3: backtest and final models
predict.py                 V3: projections, comps, SHAP drivers and the site's JSON
docs/                      V3 site (GitHub Pages)
data/final/                V3 tables, plus temp.csv for the V2 app
nba-draft-dashboard.py     the V2 Streamlit app
analysis_module/           V1 notebook (regression + archetype classification)
data_collection_module/    V1 scrapers for Tankathon, Basketball Reference and nba.com
requirements.txt           what the Streamlit app needs
requirements-model.txt     what the V3 scripts need
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
