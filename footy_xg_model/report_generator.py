"""
Unified HTML report generator for the FootyStats analytics platform.

Produces a single, self-contained HTML report that combines:
  Part 1 — Season-level analytics (team stats, top performers, goal
           prediction model, next-season projections, risk flags, breakout
           candidates).
  Part 2 — xG shot-probability model (metrics, diagnostic plots, model
           comparison, feature importance, sample predictions).

All plots are embedded as base64 PNGs so the report can be opened anywhere.
"""

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ReportData:
    """Container for every piece of data the unified report needs."""

    # -- Season-level analytics (Part 1) ------------------------------------
    num_players: int = 0
    num_teams: int = 0
    total_goals: int = 0
    total_assists: int = 0

    team_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_performers: pd.DataFrame = field(default_factory=pd.DataFrame)

    goal_model_name: str = ""
    goal_model_mae: float = 0.0
    goal_model_comparison: List[Dict[str, Any]] = field(default_factory=list)
    goal_feature_importances: Dict[str, float] = field(default_factory=dict)

    player_projections: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_projections: pd.DataFrame = field(default_factory=pd.DataFrame)

    discipline_risks: pd.DataFrame = field(default_factory=pd.DataFrame)
    breakout_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)

    next_season_games: int = 12

    # -- xG model (Part 2) -------------------------------------------------
    shots_count: int = 0
    train_size: int = 0
    test_size: int = 0
    goals_count: int = 0
    non_goals_count: int = 0

    best_model_name: str = ""
    best_model_params: Dict[str, Any] = field(default_factory=dict)

    model_comparison: List[Dict[str, Any]] = field(default_factory=list)

    metrics: Dict[str, float] = field(default_factory=dict)
    classification_report: str = ""

    sample_predictions: List[Dict[str, Any]] = field(default_factory=list)

    artifacts_dir: Path = field(default_factory=Path)

    feature_importances: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def page_background_css() -> str:
    """
    Body + hero background: optional project image with a dark overlay so text stays readable.
    Falls back to solid page color if the image is missing.
    """
    path = config.PROJECT_ROOT / "Background image.jfif"
    b64 = _img_to_base64(path)
    overlay = "rgba(13, 17, 23, 0.84)"
    if not b64:
        return """
body { background-color: #0d1117; }
"""
    url = f"data:image/jpeg;base64,{b64}"
    return f"""
html {{ min-height: 100%; }}
body {{
  background-color: #0d1117;
  background-image: linear-gradient({overlay}, {overlay}), url({url});
  background-size: cover, cover;
  background-position: center center;
  background-attachment: fixed;
  background-repeat: no-repeat;
}}
.hero {{
  background: linear-gradient(135deg, rgba(15, 42, 26, 0.92) 0%, rgba(13, 17, 23, 0.88) 100%) !important;
}}
.divider-label {{
  background: rgba(13, 17, 23, 0.92) !important;
}}
"""


def _feat_bars(importances: Dict[str, float], max_items: int = 12) -> str:
    if not importances:
        return '<p class="no-data">Feature importance data not available.</p>'
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:max_items]
    mx = max(v for _, v in items) or 1
    rows = []
    for name, val in items:
        pct = val / mx * 100 if mx else 0
        rows.append(
            f'<div class="feat-row">'
            f'<div class="feat-name">{name.replace("_", " ").title()}</div>'
            f'<div class="feat-bar"><div class="feat-fill" style="width:{pct:.0f}%"></div></div>'
            f'<div class="feat-val">{val:.3f}</div>'
            f'</div>'
        )
    return "\n".join(rows)


def _pos_badge(pos: str) -> str:
    mapping = {
        "Forward": ("FWD", "pos-Forward"),
        "Midfielder": ("MID", "pos-Midfielder"),
        "Defender": ("DEF", "pos-Defender"),
        "Goalkeeper": ("GK", "pos-Goalkeeper"),
    }
    short, cls = mapping.get(pos, (pos[:3].upper() if pos else "N/A", "pos-Midfielder"))
    return f'<span class="pos-badge {cls}">{short}</span>'


_MEDAL = {0: "medal-1", 1: "medal-2", 2: "medal-3"}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_season_overview(d: ReportData) -> str:
    return f"""
    <section>
      <h2>Season Overview</h2>
      <div class="grid-4">
        <div class="card"><div class="val">{d.num_players}</div><div class="lbl">Players Analysed</div><div class="sub">{d.num_teams} teams</div></div>
        <div class="card"><div class="val">{d.total_goals}</div><div class="lbl">Total Goals</div><div class="sub">across all teams</div></div>
        <div class="card"><div class="val">{d.total_assists}</div><div class="lbl">Total Assists</div><div class="sub">across all teams</div></div>
        <div class="card"><div class="val">&plusmn;{d.goal_model_mae:.1f}</div><div class="lbl">Goal Model Accuracy</div><div class="sub">MAE (LOO-CV)</div></div>
      </div>
    </section>"""


def _section_team_performance(d: ReportData) -> str:
    if d.team_stats.empty:
        return ""
    max_gpg = d.team_stats["team_goals_per_app"].max() or 1
    rows = ""
    for i, (_, r) in enumerate(d.team_stats.iterrows()):
        medal = _MEDAL.get(i, "")
        bar_w = int(r["team_goals_per_app"] / max_gpg * 100)
        rows += (
            f'<tr>'
            f'<td class="rank {medal}">{i+1}</td>'
            f'<td>{r["team_name"]}</td>'
            f'<td>{int(r["total_goals"])}</td>'
            f'<td>{int(r["total_assists"])}</td>'
            f'<td>{int(r["total_apps"])}</td>'
            f'<td><div class="bar-wrap"><div class="bar" style="width:{bar_w}px"></div> {r["team_goals_per_app"]:.3f}</div></td>'
            f'<td>{int(r["squad_size"])}</td>'
            f'</tr>'
        )
    return f"""
    <section>
      <h2>Team Performance</h2>
      <div class="tbl-wrap"><table>
        <thead><tr><th>#</th><th>Team</th><th>Goals</th><th>Assists</th><th>Apps</th><th>Goals / App</th><th>Squad</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </section>"""


def _section_top_performers(d: ReportData) -> str:
    if d.top_performers.empty:
        return ""
    rows = ""
    for i, (_, r) in enumerate(d.top_performers.iterrows()):
        medal = _MEDAL.get(i, "")
        gc = int(r["goals"] + r["assists"])
        rows += (
            f'<tr>'
            f'<td class="rank {medal}">{i+1}</td>'
            f'<td>{r["player_name"]}</td>'
            f'<td>{r["team_name"]}</td>'
            f'<td>{_pos_badge(r["player_position"])}</td>'
            f'<td>{int(r["appearances"])}</td>'
            f'<td>{int(r["goals"])}</td>'
            f'<td>{int(r["assists"])}</td>'
            f'<td>{gc}</td>'
            f'<td>{r["performance_score"]:.2f}</td>'
            f'</tr>'
        )
    return f"""
    <section>
      <h2>Top 10 Performers</h2>
      <div class="tbl-wrap"><table>
        <thead><tr><th>#</th><th>Player</th><th>Team</th><th>Position</th><th>Apps</th><th>Goals</th><th>Assists</th><th>G+A</th><th>Score</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </section>"""


def _section_goal_model(d: ReportData) -> str:
    model_rows = ""
    for m in d.goal_model_comparison:
        is_best = m["name"] == d.goal_model_name
        css = "model-best" if is_best else ""
        tag = ' <span class="winner-tag">SELECTED</span>' if is_best else ""
        val_css = "model-best-val" if is_best else "model-other-val"
        model_rows += (
            f'<div class="model-row {css}">'
            f'<span class="model-name">{m["name"]}{tag}</span>'
            f'<span class="model-metric {val_css}">MAE = &plusmn;{m["mae"]:.2f} goals</span>'
            f'</div>'
        )

    return f"""
    <section>
      <div class="grid-2">
        <div>
          <h2>Goal Prediction Model</h2>
          <div class="model-box">{model_rows}</div>
          <div class="note"><strong>Note:</strong> Accuracy validated using Leave-One-Out Cross-Validation &mdash; the most reliable method for small datasets.</div>
        </div>
        <div>
          <h2>Goal Model Feature Importance</h2>
          <div class="model-box">{_feat_bars(d.goal_feature_importances)}</div>
        </div>
      </div>
    </section>"""


def _section_player_projections(d: ReportData) -> str:
    if d.player_projections.empty:
        return ""
    top = d.player_projections.nlargest(10, "projected_goals").reset_index(drop=True)
    rows = ""
    for i, (_, r) in enumerate(top.iterrows()):
        medal = _MEDAL.get(i, "")
        rows += (
            f'<tr>'
            f'<td class="rank {medal}">{i+1}</td>'
            f'<td>{r["player_name"]}</td>'
            f'<td>{r["team_name"]}</td>'
            f'<td>{_pos_badge(r["player_position"])}</td>'
            f'<td>{int(r["proj_appearances"])}</td>'
            f'<td><strong>{r["projected_goals"]:.1f}</strong></td>'
            f'<td>{r["proj_assists"]:.1f}</td>'
            f'<td>{r["projected_goal_contributions"]:.1f}</td>'
            f'</tr>'
        )
    return f"""
    <section>
      <h2>Next Season Player Projections ({d.next_season_games} Games)</h2>
      <div class="tbl-wrap"><table>
        <thead><tr><th>#</th><th>Player</th><th>Team</th><th>Position</th><th>Proj Apps</th><th>Proj Goals</th><th>Proj Assists</th><th>Proj G+A</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </section>"""


def _section_team_projections(d: ReportData) -> str:
    if d.team_projections.empty:
        return ""
    rows = ""
    for i, (_, r) in enumerate(d.team_projections.iterrows()):
        medal = _MEDAL.get(i, "")
        pct = r.get("trend_pct", 0)
        arrow = "&#9650;" if pct >= 0 else "&#9660;"
        css = "risk-low" if pct >= 0 else "risk-med"
        rows += (
            f'<tr>'
            f'<td class="rank {medal}">{i+1}</td>'
            f'<td>{r["team_name"]}</td>'
            f'<td>{r["proj_goals"]:.1f}</td>'
            f'<td>{r["proj_assists"]:.1f}</td>'
            f'<td>{r["proj_goals_per_game"]:.3f}</td>'
            f'<td class="{css}">{arrow} {abs(pct):.0f}% goals</td>'
            f'</tr>'
        )
    return f"""
    <section>
      <h2>Team Projections &mdash; Next Season</h2>
      <div class="tbl-wrap"><table>
        <thead><tr><th>#</th><th>Team</th><th>Proj Goals</th><th>Proj Assists</th><th>Goals / App</th><th>Trend vs Current</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </section>"""


def _section_risk_breakout(d: ReportData) -> str:
    # Discipline risks
    if d.discipline_risks.empty:
        risk_rows = '<tr><td colspan="5" style="color:#8b949e;text-align:center">No high-risk players</td></tr>'
    else:
        risk_rows = ""
        for _, r in d.discipline_risks.iterrows():
            css = "risk-high" if r["proj_discipline"] >= 4 else "risk-med"
            level = "HIGH" if r["proj_discipline"] >= 4 else "MED"
            risk_rows += (
                f'<tr>'
                f'<td>{r["player_name"]}</td>'
                f'<td>{r["team_name"]}</td>'
                f'<td>{r["proj_yellow"]:.1f}</td>'
                f'<td>{r["proj_red"]:.1f}</td>'
                f'<td><span class="{css}">&#9679; {level} ({r["proj_discipline"]:.1f})</span></td>'
                f'</tr>'
            )

    # Breakout candidates
    if d.breakout_candidates.empty:
        bo_rows = '<tr><td colspan="5" style="color:#8b949e;text-align:center">No breakout candidates identified</td></tr>'
    else:
        bo_rows = ""
        for i, (_, r) in enumerate(d.breakout_candidates.iterrows()):
            tag = ' <span class="breakout-chip">Hottest</span>' if i == 0 else ""
            bo_rows += (
                f'<tr>'
                f'<td>{r["player_name"]}{tag}</td>'
                f'<td>{r["team_name"]}</td>'
                f'<td>{int(r["appearances"])}</td>'
                f'<td><strong class="risk-low">{r["goals_per_game"]:.2f}</strong></td>'
                f'<td>{r["projected_goals"]:.1f}</td>'
                f'</tr>'
            )

    return f"""
    <section>
      <div class="grid-2">
        <div>
          <h2>Discipline Risk Flags</h2>
          <div class="tbl-wrap"><table>
            <thead><tr><th>Player</th><th>Team</th><th>Proj Yellows</th><th>Proj Reds</th><th>Risk</th></tr></thead>
            <tbody>{risk_rows}</tbody>
          </table></div>
        </div>
        <div>
          <h2>Breakout Candidates</h2>
          <div class="tbl-wrap"><table>
            <thead><tr><th>Player</th><th>Team</th><th>Apps</th><th>Goals/Game</th><th>Proj Goals</th></tr></thead>
            <tbody>{bo_rows}</tbody>
          </table></div>
        </div>
      </div>
    </section>"""


# ---------------------------------------------------------------------------
# xG model sections (Part 2)
# ---------------------------------------------------------------------------

def _section_xg_overview(d: ReportData) -> str:
    total = d.goals_count + d.non_goals_count
    rate = d.goals_count / total * 100 if total else 0
    return f"""
    <section>
      <h2>xG Model &mdash; Dataset Overview</h2>
      <div class="grid-4">
        <div class="card"><div class="val">{d.shots_count}</div><div class="lbl">Total Shots</div><div class="sub">synthetic shot events</div></div>
        <div class="card"><div class="val">{d.train_size}</div><div class="lbl">Training Set</div><div class="sub">~{d.train_size / max(d.shots_count, 1) * 100:.0f}% of data</div></div>
        <div class="card"><div class="val">{d.test_size}</div><div class="lbl">Test Set</div><div class="sub">held out for evaluation</div></div>
        <div class="card"><div class="val">{rate:.1f}%</div><div class="lbl">Goal Rate</div><div class="sub">{d.goals_count} goals / {total} shots</div></div>
      </div>
    </section>"""


def _section_xg_metrics(d: ReportData) -> str:
    descs = {
        "log_loss": "Lower is better. Measures probabilistic prediction quality.",
        "roc_auc": "Higher is better. Area under ROC curve (0.5 = random).",
        "brier_score": "Lower is better. Mean squared error of probabilities.",
        "precision@0.5": "At threshold 0.5: proportion of predicted goals that were actual goals.",
        "recall@0.5": "At threshold 0.5: proportion of actual goals correctly predicted.",
    }
    rows = ""
    for k, v in d.metrics.items():
        desc = descs.get(k, "")
        desc_html = f'<div class="metric-desc">{desc}</div>' if desc else ""
        rows += (
            f'<tr>'
            f'<td class="metric-label">{k.replace("_", " ").title()}</td>'
            f'<td class="metric-value">{v:.4f}</td>'
            f'<td class="metric-desc-cell">{desc_html}</td>'
            f'</tr>'
        )
    return f"""
    <section>
      <h2>xG Model &mdash; Test Set Metrics</h2>
      <div class="tbl-wrap"><table>
        <thead><tr><th>Metric</th><th>Value</th><th>Description</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </section>"""


def _section_xg_model_comparison(d: ReportData) -> str:
    model_rows = ""
    for m in d.model_comparison:
        name = m.get("name", "Unknown")
        ll = m.get("validation_log_loss", float("nan"))
        is_best = name.lower().replace(" ", "_") == d.best_model_name.lower().replace(" ", "_")
        css = "model-best" if is_best else ""
        tag = ' <span class="winner-tag">SELECTED</span>' if is_best else ""
        val_css = "model-best-val" if is_best else "model-other-val"
        model_rows += (
            f'<div class="model-row {css}">'
            f'<span class="model-name">{name}{tag}</span>'
            f'<span class="model-metric {val_css}">Log Loss = {ll:.4f}</span>'
            f'</div>'
        )

    params_html = ""
    if d.best_model_params:
        params_html = "<ul class='param-list'>"
        for k, v in d.best_model_params.items():
            params_html += f"<li><strong>{k.replace('clf__', '')}</strong>: {v}</li>"
        params_html += "</ul>"

    return f"""
    <section>
      <div class="grid-2">
        <div>
          <h2>xG Model Comparison</h2>
          <div class="model-box">{model_rows}</div>
          <p style="font-size:0.8rem;color:#8b949e;margin-top:12px;">Best model selected by validation Log Loss, then calibrated with isotonic regression.</p>
        </div>
        <div>
          <h2>Best xG Model: {d.best_model_name.replace("_", " ").title()}</h2>
          <div class="model-box">
            <p style="color:#8b949e;font-size:0.9rem;">Hyperparameters (after tuning):</p>
            {params_html or "<p class='no-data'>No parameters recorded.</p>"}
          </div>
        </div>
      </div>
    </section>"""


def _section_xg_plots(d: ReportData) -> str:
    a = d.artifacts_dir
    parts = []
    for fname, title in [
        ("roc_curve.png", "ROC Curve"),
        ("precision_recall_curve.png", "Precision-Recall Curve"),
        ("confusion_matrix.png", "Confusion Matrix (threshold=0.5)"),
        ("calibration_curve.png", "Calibration Curve"),
    ]:
        b64 = _img_to_base64(a / fname)
        img = f'<img src="data:image/png;base64,{b64}" alt="{title}" class="report-img" />' if b64 else ""
        parts.append(f'<div class="plot-box"><h3>{title}</h3>{img}</div>')

    fi_b64 = _img_to_base64(a / "feature_importances.png")
    fi_img = f'<img src="data:image/png;base64,{fi_b64}" alt="Feature Importances" class="report-img" />' if fi_b64 else ""

    return f"""
    <section>
      <h2>xG Diagnostic Plots</h2>
      <div class="plot-grid">
        {parts[0]}
        {parts[1]}
        {parts[2]}
        {parts[3]}
      </div>
      <div class="plot-box" style="margin-top:20px;"><h3>Feature Importances</h3>{fi_img}</div>
    </section>"""


def _section_xg_classification_report(d: ReportData) -> str:
    return f"""
    <section>
      <h2>xG Classification Report</h2>
      <div class="pre-block">{d.classification_report or "N/A"}</div>
    </section>"""


def _section_xg_feat_importance(d: ReportData) -> str:
    return f"""
    <section>
      <h2>xG Feature Importance (Inline)</h2>
      <div class="model-box">{_feat_bars(d.feature_importances)}</div>
    </section>"""


def _section_xg_predictions(d: ReportData) -> str:
    if not d.sample_predictions:
        return ""
    rows = ""
    for i, p in enumerate(d.sample_predictions, 1):
        xg = p.get("xg", 0)
        xg_pct = int(round(xg * 100))
        rows += (
            f'<tr>'
            f'<td>{i}</td>'
            f'<td>{p.get("player_name", "N/A")}</td>'
            f'<td>({p.get("x", 0):.1f}, {p.get("y", 0):.1f})</td>'
            f'<td>{p.get("body_part", "N/A")}</td>'
            f'<td>{p.get("match_minute", "N/A")}</td>'
            f'<td><span class="xg-val">xG = {xg:.2f}</span></td>'
            f'<td><div class="xg-bar"><div class="xg-fill" style="width:{xg_pct}%"></div></div></td>'
            f'</tr>'
        )
    return f"""
    <section>
      <h2>Example xG Predictions</h2>
      <div class="tbl-wrap"><table>
        <thead><tr><th>#</th><th>Player</th><th>Location (x, y)</th><th>Body Part</th><th>Minute</th><th>xG</th><th>Visual</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </section>"""


# ---------------------------------------------------------------------------
# CSS (shared)
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Tahoma, sans-serif; color: #e6edf3; min-height: 100vh; line-height: 1.5; }
.hero { background: linear-gradient(135deg, #0f2a1a 0%, #0d1117 100%);
        border-bottom: 1px solid #30363d; padding: 40px 32px 32px; }
.hero h1 { font-size: 2rem; font-weight: 700; color: #4ade80; }
.hero p  { color: #8b949e; margin-top: 8px; font-size: 0.95rem; }
.badge   { display: inline-block; background: #22c55e22; border: 1px solid #22c55e55;
           color: #4ade80; border-radius: 20px; padding: 4px 12px; font-size: 0.8rem; margin-top: 10px; margin-right: 8px; }
main { max-width: 1200px; margin: 0 auto; padding: 32px; }
section { margin-bottom: 40px; }
h2 { font-size: 1.15rem; font-weight: 600; color: #f0f6fc; border-left: 3px solid #4ade80;
     padding-left: 12px; margin-bottom: 16px; }
.divider { border: none; border-top: 2px solid #30363d; margin: 48px 0; }
.divider-label { text-align: center; color: #4ade80; font-size: 1rem; font-weight: 600;
                 background: #0d1117; display: inline-block; padding: 0 16px;
                 position: relative; top: -12px; }
.divider-wrap { text-align: center; }

.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; }
.card .val { font-size: 2rem; font-weight: 700; color: #4ade80; }
.card .lbl { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }
.card .sub { font-size: 0.75rem; color: #6e7681; margin-top: 2px; }

table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
thead th { background: #1c2128; color: #8b949e; font-weight: 500; text-align: left;
           padding: 10px 14px; border-bottom: 1px solid #30363d; font-size: 0.78rem;
           text-transform: uppercase; letter-spacing: 0.04em; }
tbody tr { border-bottom: 1px solid #21262d; transition: background 0.15s; }
tbody tr:hover { background: #1c2128; }
tbody td { padding: 10px 14px; color: #c9d1d9; }
.tbl-wrap { background: #161b22; border: 1px solid #30363d; border-radius: 10px; overflow: hidden; }

.rank { font-weight: 700; color: #4ade80; width: 30px; }
.medal-1 { color: #ffd700; }
.medal-2 { color: #c0c0c0; }
.medal-3 { color: #cd7f32; }

.pos-badge { display: inline-block; border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; }
.pos-Forward    { background: #14532d; color: #4ade80; }
.pos-Midfielder { background: #0f2a35; color: #67e8f9; }
.pos-Defender   { background: #2a1a3a; color: #bc8cff; }
.pos-Goalkeeper { background: #3a2a1a; color: #e3b341; }

.bar-wrap { display: flex; align-items: center; gap: 10px; }
.bar { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #16a34a, #4ade80); min-width: 2px; }

.model-box { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; }
.model-row { display: flex; justify-content: space-between; align-items: center;
             padding: 10px 0; border-bottom: 1px solid #21262d; }
.model-row:last-child { border-bottom: none; }
.model-row.model-best { background: #14532d22; border-radius: 6px; margin: 0 -10px; padding: 10px; }
.model-name { color: #c9d1d9; font-size: 0.9rem; }
.model-metric { font-size: 0.85rem; }
.model-best-val { color: #4ade80; font-weight: 600; }
.model-other-val { color: #8b949e; }
.winner-tag { background: #14532d; color: #4ade80; border-radius: 6px; padding: 2px 8px; font-size: 0.72rem; margin-left: 8px; }
.param-list { margin: 10px 0 0 20px; color: #8b949e; font-size: 0.9rem; }
.param-list li { margin: 4px 0; }

.note { background: #1c2128; border: 1px solid #30363d; border-radius: 8px;
        padding: 14px 18px; font-size: 0.82rem; color: #8b949e; margin-top: 20px; line-height: 1.6; }
.note strong { color: #e3b341; }

.risk-high { color: #f85149; }
.risk-med  { color: #e3b341; }
.risk-low  { color: #4ade80; }

.breakout-chip { background: #14532d; color: #4ade80; border: 1px solid #4ade8044;
                 border-radius: 12px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; }

.report-img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #30363d; }
.plot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.plot-box { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; overflow: hidden; }
.plot-box h3 { font-size: 0.95rem; color: #8b949e; margin-bottom: 12px; }

.feat-row { display: flex; align-items: center; gap: 12px; padding: 6px 0; }
.feat-name { width: 200px; color: #c9d1d9; font-size: 0.85rem; }
.feat-bar  { flex: 1; height: 8px; border-radius: 4px; background: #21262d; }
.feat-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #4ade80, #67e8f9); }
.feat-val  { width: 56px; text-align: right; color: #8b949e; font-size: 0.8rem; }

.metric-label { color: #c9d1d9; }
.metric-value { font-weight: 600; color: #4ade80; }
.metric-desc { font-size: 0.78rem; color: #6e7681; margin-top: 2px; }
.metric-desc-cell { max-width: 280px; }

.xg-val { font-weight: 600; color: #4ade80; }
.xg-bar { height: 6px; border-radius: 3px; background: #21262d; overflow: hidden; }
.xg-fill { height: 100%; background: linear-gradient(90deg, #22c55e, #a3e635); }

.pre-block { background: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 16px;
             font-family: 'Consolas', monospace; font-size: 0.82rem; color: #c9d1d9;
             white-space: pre-wrap; overflow-x: auto; }
.no-data { color: #6e7681; font-style: italic; padding: 16px; }
footer { text-align: center; padding: 32px; color: #6e7681; font-size: 0.78rem; border-top: 1px solid #21262d; margin-top: 40px; }

@media (max-width: 900px) {
  .grid-4 { grid-template-columns: 1fr 1fr; }
  .grid-2, .plot-grid { grid-template-columns: 1fr; }
}
"""


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_html_report(data: ReportData, output_path: Path) -> None:
    """
    Generate the unified FootyStats HTML report that merges season-level
    analytics and xG model evaluation into a single self-contained file.
    """
    # Part 1 — Season analytics sections
    part1 = "".join([
        _section_season_overview(data),
        _section_team_performance(data),
        _section_top_performers(data),
        _section_goal_model(data),
        _section_player_projections(data),
        _section_team_projections(data),
        _section_risk_breakout(data),
    ])

    # Part 2 — xG model sections
    part2 = "".join([
        _section_xg_overview(data),
        _section_xg_metrics(data),
        _section_xg_model_comparison(data),
        _section_xg_plots(data),
        _section_xg_classification_report(data),
        _section_xg_feat_importance(data),
        _section_xg_predictions(data),
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FootyStats &mdash; Unified Analytics Report</title>
<style>{_CSS}
{page_background_css()}</style>
</head>
<body>
<div class="hero">
  <h1>FootyStats Analytics Platform</h1>
  <p>Combined season analytics and Expected Goals (xG) shot-probability model</p>
  <span class="badge">{data.goal_model_name} &middot; LOO-CV &middot; &plusmn;{data.goal_model_mae:.1f} goal accuracy</span>
  <span class="badge">xG: {data.best_model_name.replace("_", " ").title()} &middot; Probability Calibrated</span>
</div>
<main>

  {part1}

  <div class="divider-wrap">
    <hr class="divider">
    <span class="divider-label">Expected Goals (xG) Shot Model</span>
  </div>

  {part2}

</main>
<footer>
  FootyStats Analytics Platform &mdash; Season Analytics + xG Model &middot;
  {data.num_players} Players &middot; {data.num_teams} Teams &middot;
  {data.shots_count} Synthetic Shots
</footer>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
