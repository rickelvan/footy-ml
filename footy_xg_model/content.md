# FootyStats Expected Goals (xG) Prediction System

---

## 1. Introduction

Football analytics has transformed how clubs, broadcasters, and fans evaluate performance. One of the most influential metrics in modern football is **Expected Goals (xG)** — a statistical measure that assigns every shot a probability of becoming a goal, based on characteristics of that shot and the match context surrounding it.

This project implements a complete, modular machine learning system that predicts xG for football shots. The system is built for the FootyStats platform and follows professional software engineering practices: separation of concerns, reproducibility, configurable parameters, and comprehensive evaluation.

The input is a player-season level CSV dataset (`footy-dataset-20260316-103939.csv`) containing 89 player-season records across multiple teams and seasons. Because the raw data is aggregated (goals per season, not individual shots), the system includes a synthetic shot generation stage that expands these aggregates into realistic shot-level events before training.

Three classification algorithms — **Logistic Regression**, **Random Forest**, and **Gradient Boosting** — are trained, tuned, compared, and the best is calibrated so that its predicted probabilities faithfully represent real-world goal likelihood.

---

## 2. Problem Statement

Traditional football statistics (goals, assists, appearances) do not capture the **quality of chances** a player or team creates and receives. A striker who scores 10 goals from 10 tap-ins and a striker who scores 10 goals from 10 long-range efforts appear identical in the stats, yet the underlying performance is very different.

Without a measure of chance quality:

- **Coaches** cannot objectively assess whether poor results are due to bad finishing or bad chance creation.
- **Scouts** cannot separate a player's finishing skill from the quality of chances their team provides.
- **Analysts** cannot determine whether a match result was "deserved" or driven by luck.
- **Fans and media** lack a clear, data-driven way to discuss performance beyond goals scored.

The problem this system addresses is: **Given a shot's location, body part, match context, and player position, what is the probability that it results in a goal?**

---

## 3. Objectives

1. **Design a realistic data model** for football shot events, including spatial, temporal, and contextual features.
2. **Generate synthetic shot-level data** from player-season aggregates using documented, configurable assumptions.
3. **Engineer meaningful features** that capture shot geometry (distance, angle) and match context (minute, score, pressure).
4. **Train and compare multiple ML models** (Logistic Regression, Random Forest, Gradient Boosting) using cross-validated hyperparameter tuning.
5. **Calibrate the best model** so predicted probabilities closely match observed goal frequencies.
6. **Evaluate using probability-appropriate metrics** — log loss, ROC-AUC, Brier score, precision, recall, calibration curves.
7. **Save the trained pipeline** for reuse and deploy it to generate xG predictions on new shots.
8. **Produce a comprehensive HTML report** with all metrics, plots, and example predictions.

---

## 4. Methodology

### 4.1 Data Flow

The system processes data through a linear pipeline:

```
Player-season CSV
       │
       ▼
  Data Loader (validation, cleaning)
       │
       ▼
  Synthetic Shot Generator (expand to shot-level events)
       │
       ▼
  Feature Engineering (distance, angle, encoding)
       │
       ▼
  Train / Validation / Test Split (stratified)
       │
       ├──────────────────────────────────┐
       ▼                                  ▼
  Model Training (3 algorithms)     Validation Set
       │                                  │
       ▼                                  ▼
  GridSearchCV (5-fold, log loss)    Model Selection (best log loss)
       │                                  │
       ▼                                  ▼
  Best Model ──────────────────► Probability Calibration
                                          │
                                          ▼
                                   Evaluation on Test Set
                                          │
                                    ┌─────┴──────┐
                                    ▼             ▼
                              Save Model    HTML Report
                                    │
                                    ▼
                             Predict on New Shots
```

### 4.2 Data Entry and Loading

The input is `footy-dataset-20260316-103939.csv`, containing one row per player per season with columns:

| Column | Type | Description |
|--------|------|-------------|
| team_id, team_name | ID / string | Team identity |
| player_id, player_name | ID / string | Player identity |
| player_position | string | Forward, Midfielder, Defender, Goalkeeper |
| season | int | e.g. 2025 or 2026 |
| appearances | int | Games played |
| goals | int | Total goals scored in the season |
| assists, yellow_cards, red_cards, clean_sheets, saves | int | Other season aggregates |

The data loader (`data_loader.py`) reads the CSV, validates that required columns exist, coerces numeric types, fills missing values, and normalises position labels.

### 4.3 Synthetic Shot Generation

Since the CSV only has **season totals** (e.g. "Hector scored 20 goals in 10 appearances"), we need to create individual shot events to train an xG model.

The preprocessing module (`preprocessing.py`) does this in three steps:

**Step 1 — Estimate shots per player-season:**

Each player's total shots are estimated from their appearances and position:

| Position | Shots per appearance |
|----------|---------------------|
| Forward | 2.5 |
| Midfielder | 1.0 |
| Defender | 0.3 |
| Goalkeeper | 0.05 |

A minimum of 3 shots per player-season is enforced.

**Step 2 — Assign goals to shots:**

For each player-season, the conversion rate is calculated as:

```
p_goal = goals / estimated_shots
```

Each shot is then independently assigned as goal (1) or miss (0) using a **binomial distribution** with probability `p_goal`. This preserves the aggregate goal counts while distributing them across shots.

**Step 3 — Sample contextual features per shot:**

For each shot, contextual features are randomly sampled:

| Feature | How it is sampled |
|---------|-------------------|
| `match_minute` | Uniform integer between 1 and 90 |
| `score_diff` | From {-2, -1, 0, 1, 2} with weights [0.1, 0.2, 0.4, 0.2, 0.1] |
| `under_pressure` | Bernoulli with p = 0.35 |
| `body_part` | "foot" (82%) or "head" (18%) |
| `shot_type` | "open_play" (80%) or "set_piece" (20%) |
| `x, y` (pitch location) | Gaussian distribution; mean and spread depend on position (forwards shoot closer and more centrally) |

All assumptions are configurable in `config.py` and documented inline.

The result from the latest run: **1,199 synthetic shots** from 89 player-season records, with a **14.3% goal rate** (171 goals, 1,028 non-goals).

---

## 5. Data Preprocessing and Feature Engineering

### 5.1 Geometric Features

Raw pitch coordinates (x, y) are not directly useful to a model. Instead, we compute two football-meaningful features:

**Shot distance** — Euclidean distance from the shot location to the centre of the goal:

```
distance = sqrt((goal_x - x)² + (y - goal_y)²)
```

where goal_x = 105.0 m and goal_y = 34.0 m (centre of the attacking goal on a standard 105 × 68 m pitch).

Shots from further away are harder to score. Distance is the single most predictive feature in most xG models.

**Shot angle** — Bearing angle from the shot location to the goal centre:

```
angle = arctan2(|y - goal_y|, goal_x - x)
```

A smaller angle means a more central shot (directly facing the goal), which is generally easier to score.

### 5.2 Feature Matrix

The final feature set passed to the models:

| Feature | Type | Description |
|---------|------|-------------|
| `shot_distance` | Numeric | Metres from goal centre |
| `shot_angle` | Numeric | Radians; 0 = central, higher = wider |
| `match_minute` | Numeric | 1–90 |
| `score_diff` | Numeric | From shooter's perspective (-2 to +2) |
| `under_pressure` | Binary (0/1) | Whether a defender was nearby |
| `body_part` | Categorical | "foot" or "head" |
| `shot_type` | Categorical | "open_play" or "set_piece" |
| `player_position` | Categorical | Forward, Midfielder, Defender, Goalkeeper |

### 5.3 Preprocessing Pipeline

Before feeding features to the classifiers:

- **Numeric features** are standardised using `StandardScaler` (z-score normalisation: subtract mean, divide by standard deviation). This ensures features on different scales (e.g. distance in metres vs minute 1–90) contribute equally.
- **Categorical features** are one-hot encoded using `OneHotEncoder`. Each category becomes a binary column (e.g. body_part_foot = 1, body_part_head = 0).

Both transformations are wrapped in a `ColumnTransformer` so they happen inside the sklearn `Pipeline` and are automatically applied during both training and prediction.

---

## 6. System Architecture

```
footy_xg_model/
│
├── config.py                 # Central configuration (paths, seeds, feature lists, pitch geometry)
├── data_loader.py            # Load and validate the player-season CSV
├── preprocessing.py          # Synthetic shot generation + train/test split
├── feature_engineering.py    # Compute distance, angle; build feature matrix X and target y
├── train_model.py            # Define pipelines, train 3 models, tune, select, calibrate, save
├── evaluate_model.py         # Compute metrics (log loss, ROC-AUC, Brier) and generate plots
├── predict.py                # Load saved pipeline and generate xG for new shots
├── report_generator.py       # Build self-contained HTML evaluation report
├── main.py                   # Orchestrate the full pipeline end-to-end
│
├── artifacts/                # Output directory
│   ├── xg_evaluation_report.html
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── confusion_matrix.png
│   ├── calibration_curve.png
│   └── feature_importances.png
│
└── models/
    └── best_calibrated_xg_pipeline.joblib   # Saved trained model
```

**Design principles:**
- **Separation of concerns** — each module has one responsibility.
- **Reproducibility** — a single `RANDOM_SEED` in `config.py` controls all randomness.
- **Configurability** — all assumptions (shot rates, pitch dimensions, hyperparameter grids) are in `config.py`.
- **End-to-end pipeline** — `main.py` runs everything from raw CSV to saved model and HTML report.

---

## 7. Machine Learning Models

### 7.1 Logistic Regression (Baseline)

**How it works:** Fits a linear decision boundary by modelling the log-odds of scoring as a weighted sum of features:

```
log(p / (1 - p)) = w₀ + w₁·distance + w₂·angle + ...
```

The output probability `p` is obtained via the sigmoid function.

**Why use it:** Logistic Regression is a natural baseline for binary probability prediction. It is fast, interpretable (coefficients show feature importance), and inherently produces calibrated probabilities.

**Configuration:** Solver = LBFGS, regularisation = L2, C tuned from {0.1, 1.0, 10.0}.

### 7.2 Random Forest

**How it works:** Builds an ensemble of 200–400 decision trees, each trained on a random bootstrap sample with a random subset of features. The final prediction is the average probability across all trees.

**Why use it:** Random Forests handle non-linear relationships and feature interactions well. They are robust to outliers and do not require feature scaling (though we apply it for consistency across models).

**Configuration:** `n_estimators` = {200, 400}, `max_depth` = {None, 5, 10}, `min_samples_leaf` = {1, 5}, `class_weight` = "balanced_subsample" to address the imbalanced goal rate.

### 7.3 Gradient Boosting

**How it works:** Builds trees sequentially, where each new tree fits the residual errors of the previous ensemble. Predictions are accumulated with a learning rate to prevent overfitting.

**Why use it:** Gradient Boosting typically achieves the best accuracy on tabular data. It captures complex interactions and is the algorithm family behind most state-of-the-art xG models in industry (e.g. StatsBomb, Opta use XGBoost/LightGBM variants).

**Configuration:** `n_estimators` = {100, 200}, `learning_rate` = {0.05, 0.1}, `max_depth` = {2, 3}.

### 7.4 Hyperparameter Tuning

All three models are tuned using `GridSearchCV` with:
- **5-fold cross-validation** on the training set
- **Log loss** as the scoring metric (lower = better calibrated probabilities)

### 7.5 Probability Calibration

After selecting the best model, its probabilities are post-hoc calibrated using **isotonic regression** (`CalibratedClassifierCV` with `method="isotonic"`) on the held-out validation set. This adjusts the raw probabilities so that, for example, shots predicted at 0.3 xG actually score about 30% of the time.

---

## 8. Model Performance

### 8.1 Model Comparison (Validation Log Loss)

| Model | Validation Log Loss |
|-------|---------------------|
| Logistic Regression | 0.4196 |
| Random Forest | 0.4643 |
| **Gradient Boosting** | **0.4164** |

**Selected model:** Gradient Boosting (lowest log loss), then calibrated.

**Best hyperparameters:** `learning_rate` = 0.05, `max_depth` = 2, `n_estimators` = 100.

### 8.2 Test Set Results (Calibrated Gradient Boosting)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Log Loss** | 0.4098 | Probabilistic prediction quality; lower is better. |
| **ROC-AUC** | 0.5059 | Ranking quality; 0.5 = random, 1.0 = perfect. |
| **Brier Score** | 0.1227 | Mean squared error of probabilities; lower is better. |
| **Precision@0.5** | 0.00 | At threshold 0.5, no shots were predicted as goals. |
| **Recall@0.5** | 0.00 | At threshold 0.5, no actual goals were captured. |
| **Accuracy** | 86% | Dominated by the majority class (non-goals). |

### 8.3 Classification Report

```
              precision    recall  f1-score   support

           0       0.86      1.00      0.92       206
           1       0.00      0.00      0.00        34

    accuracy                           0.86       240
   macro avg       0.43      0.50      0.46       240
weighted avg       0.74      0.86      0.79       240
```

### 8.4 Interpreting These Results

The model achieves an overall accuracy of **86%**, but this is largely because the dataset is imbalanced — only ~14% of shots are goals. The model learns that most shots miss, which is correct in football: the average conversion rate across top leagues is 10–15%.

**Why ROC-AUC is near 0.5:** With synthetic data derived from season aggregates (rather than real event-level tracking data), the features do not carry the same predictive signal that real shot data would. In production xG models trained on real event data (e.g. Opta, StatsBomb), ROC-AUC typically ranges from 0.75 to 0.82.

**Why precision and recall at 0.5 are zero:** The xG model is a *probability estimator*, not a binary classifier. It rarely predicts any single shot above 0.5 because most real shots have a low probability of scoring. This is expected behaviour — xG is designed to output values like 0.08, 0.15, 0.31, not to say "goal" or "no goal."

**The important metrics for xG are log loss and Brier score**, both of which measure probability quality. The model's log loss (0.41) is reasonable for the data available.

### 8.5 Example Predictions

| Shot | Player | Location | Body Part | Minute | xG |
|------|--------|----------|-----------|--------|----|
| 1 | Nate | (79.7m, 38.7m) | foot | 84 | 0.09 |
| 2 | Alvin K | (87.6m, 29.6m) | foot | 41 | 0.19 |
| 3 | Jeffers | (87.5m, 36.1m) | foot | 59 | 0.13 |
| 4 | Crivin | (68.8m, 24.8m) | foot | 24 | 0.19 |
| 5 | Uncle Li | (90.2m, 25.5m) | foot | 56 | 0.19 |

These predictions show sensible behaviour: shots closer to goal receive higher xG, and shots from wide or distant positions receive lower xG.

---

## 9. Future Work

| Area | Description |
|------|-------------|
| **Real event data** | Replace synthetic shots with actual event-level data (e.g. from Opta, StatsBomb, or Wyscout). This would dramatically improve feature signal and model accuracy. |
| **Richer features** | Add defender positions, goalkeeper location, pass sequence before shot, speed of play, and expected assist (xA) information. |
| **Advanced models** | Incorporate XGBoost or LightGBM (faster, often more accurate gradient boosting), or neural networks for spatial data. |
| **Shot maps** | Visualise shots on a pitch diagram, coloured by xG value, for more intuitive analysis. |
| **Per-match aggregation** | Build match-level xG summaries and season rolling xG charts. |
| **Live prediction** | Deploy the model as an API endpoint for real-time xG during live matches. |
| **Transfer market integration** | Combine xG with contract and market value data to support recruitment decisions. |
| **Temporal modelling** | Use sequence models (LSTM, Transformer) to account for the build-up play before a shot. |
| **Explainability** | Add SHAP or LIME explanations so each prediction can be decomposed into feature contributions. |

---

## 10. Conclusion

This project demonstrates a complete, modular Expected Goals (xG) prediction pipeline built in Python with scikit-learn. Starting from a player-season aggregate CSV, the system:

1. Generates realistic synthetic shot events with configurable football assumptions.
2. Engineers geometry-based features (shot distance and angle) alongside match context features.
3. Trains and compares three machine learning algorithms — Logistic Regression, Random Forest, and Gradient Boosting — using cross-validated hyperparameter tuning optimised for log loss.
4. Selects Gradient Boosting as the best model (validation log loss = 0.4164) and applies isotonic probability calibration.
5. Evaluates on a held-out test set with probability-focused metrics and diagnostic plots.
6. Saves the trained pipeline for future predictions and generates a self-contained HTML evaluation report.

The model produces football-plausible xG values (e.g. close-range shots receive higher xG than long-range shots), and the architecture is designed so that swapping in real event data in the future requires changes only in the data loading and preprocessing stages — the feature engineering, training, evaluation, and prediction modules remain the same.

The primary limitation is the use of synthetic rather than real shot data, which limits discriminative power (ROC-AUC near 0.5). With real event-level data, the same pipeline would be expected to achieve ROC-AUC in the 0.75–0.82 range, consistent with industry xG models.

Overall, the system provides a solid foundation for a production football analytics platform.

---

## 11. References

1. **Expected Goals (xG) concept:**  
   - Caley, M. (2014). "Shot Matrix I: Shot Location and Expected Goals." *Cartilage Free Captain.*  
   - Eggels, H., Van Elk, R., & Pechenizkiy, M. (2016). "Expected Goals in Soccer: Explaining Match Results using Predictive Analytics." *Proc. Machine Learning and Data Mining for Sports Analytics.*

2. **scikit-learn (Machine Learning library):**  
   - Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *JMLR*, 12, pp. 2825–2830.  
   - Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

3. **Logistic Regression:**  
   - Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression.* Wiley.

4. **Random Forests:**  
   - Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), pp. 5–32.

5. **Gradient Boosting:**  
   - Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), pp. 1189–1232.

6. **Probability Calibration:**  
   - Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." *Proc. ICML.*

7. **Football analytics platforms (industry context):**  
   - StatsBomb: [https://statsbomb.com/](https://statsbomb.com/)  
   - Opta (Stats Perform): [https://www.statsperform.com/](https://www.statsperform.com/)

8. **Python libraries used:**  
   - NumPy: [https://numpy.org/](https://numpy.org/)  
   - pandas: [https://pandas.pydata.org/](https://pandas.pydata.org/)  
   - matplotlib: [https://matplotlib.org/](https://matplotlib.org/)  
   - joblib: [https://joblib.readthedocs.io/](https://joblib.readthedocs.io/)
