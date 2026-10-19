# nba_draft

Projecting how NBA draft prospects will turn out, using their college and pre-draft numbers.

**Live app:** https://statosphere-nba-draft-projections.streamlit.app/

## What it does

I started this before the 2021 draft and came back to it in 2024.

**V1 (2021).** I scraped draft classes from Tankathon and advanced stats from Basketball Reference, matched the players up and trained regression models to predict each prospect's peak VORP in the NBA. I also tried classifying player archetypes, which didn't work well enough to use. The whole process is in `analysis_module/draft projections.ipynb`, and I wrote it up here: [NBA Draft 2021: an analysis](https://shoobydo.substack.com/p/nba-draft-2021-an-analysis).

**V2 (2024).** I changed what the model predicts to two ratings, both scaled from -10 to 10:

- **Peak Impact Rating**: the player's best 3-season rolling VORP.
- **Immediate Impact Rating**: how they do in their first 2 seasons.

For player comps, I clustered past draftees and used cosine similarity to pick a ceiling and a floor comparison for each prospect. The app plots those two players' ratings season by season. V2 covers the 2023 and 2024 classes, and its results are in `data/final/temp.csv`.

## Repo layout

```
nba-draft-dashboard.py     the Streamlit app
data/final/temp.csv        V2 projections that the app reads
analysis_module/           V1 notebook (regression + archetype classification)
data_collection_module/    scrapers for Tankathon, Basketball Reference and nba.com
requirements.txt           what the app needs
requirements-dev.txt       extra packages for the scrapers and the notebook
.devcontainer/             Codespaces setup that starts the app
```

## Running the app

From the repo root:

```
pip install -r requirements.txt
streamlit run nba-draft-dashboard.py
```

It opens on http://localhost:8501. You can sort by pick or either rating, filter by draft year and search for a player.

## Data sources

- [Tankathon](https://www.tankathon.com/past_drafts): draft results, measurements and college stats
- [Basketball Reference](https://www.basketball-reference.com/): NBA advanced stats, including VORP
- [nba.com/stats](https://www.nba.com/stats): player tracking stats

A few notes on the older code:

- The notebook expects the scraped CSVs in `data/`, and those aren't in the repo. You'd have to run the scrapers again to re-run it, and the sites have changed since 2021.
- The nba.com scraper was never finished.
- Run the scrapers and the notebook from the repo root too, since they use relative paths.

More of my projects: https://statxsphere.github.io/#projects
