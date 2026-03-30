# Algorithms and Functions — FootyStats xG Model

Technical documentation of the algorithms, functions, and libraries used in the xG prediction system.

---

## 1. Pipeline Overview

```
Raw CSV → Data Loader → Preprocessing (synthetic shots) → Feature Engineering
    → Train/Test Split → Model Training (3 algorithms) → Calibration → Evaluation → Save
```

---

## 2. Data Loading

### Module: `data_loader.py`

| Function | Purpose |
|----------|---------|
| `load_player_season_data(csv_path)` | Loads the player-season CSV, validates columns, coerces dtypes, fills missing values, normalises position labels. |

**Libraries:** `pandas` (read_csv, to_numeric)

---

## 3. Preprocessing

### Module: `preprocessing.py`

#### Functions

| Function | Purpose |
|----------|---------|
| `_estimate_shots_for_row(row)` | Heuristic: estimates number of shots per player-season from appearances and position (forward/midfielder/defender/goalkeeper). |
| `_simulate_shot_outcomes(num_shots, goals, rng)` | Simulates which shots were goals using a binomial distribution with p = goals / num_shots. |
| `generate_synthetic_shots(player_season_df)` | Expands player-season rows into shot-level records with sampled context (minute, score_diff, pressure, body_part, shot_type, x, y, is_goal). |
| `train_test_split_shots(shots_df, test_size)` | Stratified train/test split on `is_goal`. |

#### Algorithms Used

| Algorithm | Where | Description |
|-----------|-------|-------------|
| **Binomial sampling** | `_simulate_shot_outcomes` | `np.random.binomial(1, p_goal, size=num_shots)` — assigns goals to shots to match season totals. |
| **Gaussian sampling** | `generate_synthetic_shots` | Shot locations (x, y) sampled from normal distributions; mean and scale depend on position. |
| **Stratified split** | `train_test_split_shots` | `sklearn.model_selection.train_test_split(..., stratify=is_goal)` — keeps goal rate similar in train and test. |

**Libraries:** `numpy`, `pandas`, `sklearn.model_selection.train_test_split`

---

## 4. Feature Engineering

### Module: `feature_engineering.py`

#### Functions

| Function | Purpose |
|----------|---------|
| `compute_shot_distance_and_angle(x, y, goal_x, goal_y)` | Computes Euclidean distance and bearing angle from shot location to goal centre. |
| `add_geometric_features(shots_df)` | Adds `shot_distance` and `shot_angle` columns to the DataFrame. |
| `build_feature_target_matrices(shots_df)` | Returns feature matrix X (numeric + categorical columns) and target y (`is_goal`). |

#### Mathematical Formulas

**Shot distance (Euclidean):**
```
distance = sqrt((goal_x - x)² + (y - goal_y)²)
```

**Shot angle (bearing, radians):**
```
angle = arctan2(|y - goal_y|, goal_x - x)
```
Larger angle = more central shot = generally higher xG.

**Feature columns:**
- **Numeric:** `shot_distance`, `shot_angle`, `match_minute`, `score_diff`, `under_pressure`
- **Categorical:** `body_part`, `shot_type`, `player_position`

**Libraries:** `math`, `numpy`, `pandas`

---

## 5. Model Training

### Module: `train_model.py`

#### Preprocessing (sklearn Pipeline)

| Component | Class | Purpose |
|-----------|-------|---------|
| **Numeric transformer** | `StandardScaler()` | Z-score normalisation: (x - mean) / std. |
| **Categorical transformer** | `OneHotEncoder(handle_unknown="ignore")` | Converts categories to binary columns. |
| **Combined** | `ColumnTransformer` | Applies each transformer to the appropriate columns. |

#### Classification Algorithms

| Algorithm | sklearn Class | Role |
|-----------|---------------|------|
| **Logistic Regression** | `LogisticRegression` | Baseline probabilistic classifier. |
| **Random Forest** | `RandomForestClassifier` | Ensemble of decision trees with bagging. |
| **Gradient Boosting** | `GradientBoostingClassifier` | Sequential ensemble; each tree fits residuals. |

#### Logistic Regression Details

- **Solver:** `lbfgs` (Limited-memory BFGS)
- **Regularisation:** L2 (ridge)
- **Hyperparameters tuned:** `C` (inverse regularisation strength: 0.1, 1.0, 10.0)

#### Random Forest Details

- **Ensemble:** 200–400 trees (tuned)
- **Splitting:** Gini impurity
- **Class balancing:** `class_weight="balanced_subsample"`
- **Hyperparameters tuned:** `n_estimators`, `max_depth`, `min_samples_leaf`

#### Gradient Boosting Details

- **Loss:** Deviance (log-loss for classification)
- **Hyperparameters tuned:** `n_estimators` (100, 200), `learning_rate` (0.05, 0.1), `max_depth` (2, 3)

#### Hyperparameter Tuning

| Component | Class | Purpose |
|-----------|-------|---------|
| **Grid search** | `GridSearchCV` | Exhaustive search over parameter grid. |
| **Scoring** | `make_scorer(log_loss, response_method="predict_proba", greater_is_better=False)` | Optimises log loss (lower is better). |
| **Cross-validation** | 5-fold CV | Estimates performance and reduces overfitting. |

#### Probability Calibration

| Component | Class | Purpose |
|-----------|-------|---------|
| **Calibrator** | `CalibratedClassifierCV` | Post-hoc calibration of predicted probabilities. |
| **Method** | `isotonic` | Non-parametric monotonic regression; better for complex shapes. |
| **cv** | `prefit` | Uses already-trained base classifier; no refit. |

Calibration makes predicted probabilities better match observed goal rates (e.g. shots predicted at 0.3 xG should score ~30% of the time).

#### Functions

| Function | Purpose |
|----------|---------|
| `_build_preprocessor()` | Creates the `ColumnTransformer` for numeric and categorical features. |
| `_build_model_grids()` | Returns list of (model_name, pipeline, param_grid) for each algorithm. |
| `train_models(train_df)` | Fits all three models with `GridSearchCV`; returns `TrainedModelBundle` per model. |
| `select_best_model(models, X_val, y_val)` | Picks the model with lowest validation log loss. |
| `calibrate_model(bundle, X_val, y_val)` | Wraps the best classifier in `CalibratedClassifierCV` and returns a full pipeline. |
| `save_model(pipeline, path)` | Saves the pipeline to disk with `joblib.dump`. |

**Libraries:** `sklearn.linear_model`, `sklearn.ensemble`, `sklearn.preprocessing`, `sklearn.compose`, `sklearn.pipeline`, `sklearn.model_selection`, `sklearn.calibration`, `joblib`

---

## 6. Evaluation

### Module: `evaluate_model.py`

#### Metrics

| Metric | Formula / Meaning |
|--------|-------------------|
| **Log Loss** | −(1/n) Σ [y·log(p) + (1−y)·log(1−p)] — penalises overconfident wrong predictions. |
| **ROC-AUC** | Area under the ROC curve; measures ranking quality (0.5 = random, 1.0 = perfect). |
| **Brier Score** | Mean squared error of probabilities: (1/n) Σ (p − y)². |
| **Precision@0.5** | TP / (TP + FP) at threshold 0.5. |
| **Recall@0.5** | TP / (TP + FN) at threshold 0.5. |

#### Diagnostic Plots

| Plot | Function / Data | Purpose |
|------|-----------------|---------|
| **ROC curve** | `roc_curve`, `roc_auc_score` | True positive rate vs false positive rate at various thresholds. |
| **Precision–Recall curve** | `PrecisionRecallDisplay.from_predictions` | Precision vs recall. |
| **Confusion matrix** | `confusion_matrix`, `ConfusionMatrixDisplay` | TP, FP, TN, FN at threshold 0.5. |
| **Calibration curve** | `calibration_curve` | Predicted probability bins vs observed goal rate. |
| **Feature importance** | `model.feature_importances_` or `model.coef_` | Relative importance of each feature. |

#### Functions

| Function | Purpose |
|----------|---------|
| `evaluate_probabilistic_predictions(y_true, y_proba)` | Computes log loss, ROC-AUC, Brier score, precision, recall. |
| `plot_diagnostics(y_true, y_proba, feature_names, model, ...)` | Generates ROC, PR, confusion matrix, calibration, and feature importance plots. |
| `evaluate_on_dataframe(pipeline, df)` | Evaluates a pipeline on a DataFrame; returns metrics dict and classification report. |

**Libraries:** `sklearn.metrics`, `sklearn.calibration`, `matplotlib`

---

## 7. Prediction

### Module: `predict.py`

#### Functions

| Function | Purpose |
|----------|---------|
| `load_trained_pipeline(path)` | Loads the saved pipeline from disk with `joblib.load`. |
| `predict_xg_for_shots(pipeline, shots_df)` | Adds geometric features, builds X, returns `predict_proba(X)[:, 1]` (probability of goal). |
| `pretty_print_predictions(shots_df, xg)` | Prints a formatted table of shots and their xG values. |

**Libraries:** `joblib`, `pandas`, `numpy`

---

## 8. Config and Constants

### Module: `config.py`

| Constant | Value | Purpose |
|----------|-------|---------|
| `RANDOM_SEED` | 42 | Reproducibility. |
| `PITCH_LENGTH` | 105.0 m | Goal-to-goal distance. |
| `PITCH_WIDTH` | 68.0 m | Pitch width. |
| `TEST_SIZE` | 0.2 | 80% train, 20% test. |
| `CV_FOLDS` | 5 | Cross-validation folds for tuning. |

---

## 9. Summary Table

| Stage | Main algorithms / functions |
|-------|----------------------------|
| **Data** | pandas CSV read, dtype coercion |
| **Preprocessing** | Binomial sampling, Gaussian sampling, stratified split |
| **Features** | Euclidean distance, arctan2 angle, ColumnTransformer |
| **Numeric scaling** | StandardScaler (z-score) |
| **Categorical encoding** | OneHotEncoder |
| **Classification** | Logistic Regression, Random Forest, Gradient Boosting |
| **Tuning** | GridSearchCV, 5-fold CV, log loss scoring |
| **Calibration** | CalibratedClassifierCV (isotonic) |
| **Evaluation** | Log loss, ROC-AUC, Brier score, precision, recall, calibration curve |
| **Persistence** | joblib (save/load pipeline) |
