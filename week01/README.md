# TitanicEDA

# Project 1 — EDA on a Real Dataset: Titanic
## Dataset source

- **Dataset:** Titanic passenger manifest, the 891-row training extract used by
  Kaggle's ["Titanic — Machine Learning from Disaster"](https://www.kaggle.com/competitions/titanic)
  competition records of the RMS Titanic's 15 April 1912 voyage.
  
- **Loaded from:** a public GitHub mirror of the same file —
  `https://www.kaggle.com/competitions/titanic/data?select=train.csv`
  
- **Licence:** Kaggle distributes this dataset for the competition under its
  standard competition data terms (educational / competition use). The
  `datasciencedojo/datasets` GitHub mirror used for the reproducible URL does not
  itself impose additional terms beyond that. **Before reusing this data outside
  coursework** (e.g. any public or commercial redistribution), verify the current
  licence terms directly on the Kaggle competition page — this repo is used here for
  educational analysis only, consistent with that intent.
  
- **Collected:** the manifest describes the ship's actual 15 April 1912 voyage; the
  Kaggle competition itself has been running since 2012 and periodically re-serves

## What's in this folder

```
week01/
  project1_eda.ipynb   the analysis notebook (R1–R7)
  FINDINGS.md           the written findings document (R5)
  README.md             this file
  train_titanic.csv     required csv file of titanic
```

## How to run
1. Requirements: Python 3.10+, with `pandas`, `numpy`, `matplotlib`, `seaborn`
2. Open `project1_eda.ipynb` in Colab Notebook and **Connect & Run All**. Create folder `/content/TitanicEDA` and keep file train_titanic.csv in `TitanicEDA` folder.
3. The notebook runs top-to-bottom cleanly in under 10 seconds on a fresh kernel (verified before submission).

## Notes on the data itself

- One row = one passenger with a known survival out. Total there are 891 rows.
- The dataset is intentionally messy: ~20% of `Age` is missing, ~77% of `Cabin` is missing, and a small sentinel-like group of 15 zero-fare passengers turns out to be a historical anomaly rather than a data-entry error.
- Full detail in `FINDINGS.md`.
