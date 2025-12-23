---
layout: post
title: "AI Incident Forecasting: Repository Readme"
date: 2025‑12-23 14:00:00 +0000
categories: forecasting incidents
---
# AI Incident Forecasting

This repository contains code for forecasting trajectories of AI-related incident counts, developed during the [Apart AI Forecasting Hackathon](https://apartresearch.com/sprints/the-ai-forecasting-hackathon-2025-10-31-to-2025-11-02). We address three research questions: (RQ1) Forecasting future trajectories of total and category-level AI incident counts; (RQ2) Comparing one-year-ahead forecast accuracy to a naive baseline; and (RQ3) Assessing calibration of prediction intervals. The pipeline is implemented in Python and includes data loaders, forecasting utilities, and evaluation modules. Data are loaded from incident dataset (`classifications_MIT.csv` and `incidents.csv`), merged by incident ID, and normalized by date using the provided `data_loaders.py` functions.

## Methodology

The core forecasting pipeline uses a Poisson regression (a generalized linear model) to project annual incident totals, and a multinomial logistic regression (with spline-transformed time features) to model category shares. The forecast combines total-count and category-share models to allocate predicted incidents by category. Uncertainty is quantified via bootstrapped simulations: we resample model parameters to generate predictive distributions, from which we derive prediction intervals (e.g. 90% intervals) for total and category counts. The project computes one-year-ahead forecasts for each year in the hold-out period and compares them to a naive baseline (which uses the previous 3-year average plus average growth). Finally, we evaluate interval calibration by measuring the empirical coverage of prediction intervals against nominal confidence levels.

## Repository Content

* Data Loaders (`data_loaders.py`) – Functions to load the MIT classification and incident CSV files and merge them by incident ID (see `load_and_merge()`).
* Forecasting Helpers (`forecast_helpers.py`) – Implements the forecasting models and simulations. The ForecastConfig class (with defaults) controls parameters such as spline knots, bootstrap iterations, and target years. Key imports include `sklearn.linear_model.PoissonRegressor` and `LogisticRegression` for model fitting.
* Forecast Evaluation (`forecast_eval.py`) – Routines for backtesting and computing accuracy metrics. This includes the `backtest_forecast` function for one-step-ahead forecasting, and naive_baseline_forecast as a baseline comparator. Prediction-interval coverage is computed by checking actual counts against model-derived intervals.
* Notebooks (Jupyter) – Four notebooks illustrate the project workflow:
  * `exploration.ipynb`: Exploratory data analysis of the incident dataset (summary statistics, value counts, basic plots).
  * `forecasting_by_column.ipynb`: Core forecasting workflow applied across multiple taxonomies (e.g. “Risk Domain”, “Actor”). It demonstrates loading data, configuring ForecastConfig, and running forecast_by_category to generate forecasts and plots.
  * `model_evaluation.ipynb`: Implements one-year-ahead backtesting. It iterates over test years, fits models on training data, and records forecast vs actual counts and errors. This notebook computes MAE, MAPE, RMSE for each year and compares results to the naive baseline.
  * `calibration_analysis.ipynb`: Analyzes empirical coverage of prediction intervals. It varies uncertainty parameters and bootstraps to show how often actual counts fall inside the forecast intervals. Coverage vs. nominal confidence plots are generated to assess calibration.

## Installation

Install a recent Python 3 interpreter (3.8+ recommended). Clone this repository and install required packages. For example:

```python
git clone https://github.com/<your-org>/ai-incident-forecasting.git
cd ai-incident-forecasting
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
```

Optionally install scienceplots for styled figures (used in the notebooks).
Ensure that the data files classifications_MIT.csv and incidents.csv are placed in a data/ subdirectory, matching the paths expected by load_and_merge(). (By default, the loaders look in ../data/ relative to the scripts.)

## Usage

1. Run Notebooks: Launch Jupyter Lab or Notebook. Open and execute the notebooks in sequence. Each notebook contains commentary and code cells illustrating the analysis steps.
2. Exploration: Start with `exploration.ipynb` to inspect the data (structure, time span, missing values, etc.).
3. Forecasting: Use `forecasting_by_column.ipynb` to run forecasts. Adjust ForecastConfig settings (e.g. forecast horizon, spline degrees) as needed. This notebook produces plots of projected total counts and category breakdowns.
4. Evaluation: Use `model_evaluation.ipynb` to perform backtesting. This will output year-by-year forecasts, actual counts, error metrics, and a comparison table against the naive baseline.
5. Calibration: Finally, run `calibration_analysis.ipynb` to verify interval coverage. This notebook generates calibration curves and reports empirical coverage rates.

The analysis was conducted in accordance with the Apart Hackathon objectives. All results (plots, tables) are reproducible using the provided code and data.

## Citation

The Incident dataset:

```bibtex
@misc{mcgregor2020preventingrepeatedrealworld,
      title={Preventing Repeated Real World AI Failures by Cataloging Incidents: The AI Incident Database}, 
      author={Sean McGregor},
      year={2020},
      eprint={2011.08512},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2011.08512}, 
}
```
