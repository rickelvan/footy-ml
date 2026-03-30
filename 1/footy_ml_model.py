import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & clean ──────────────────────────────────────────────────────────
df = pd.read_csv('footy-dataset-20260204-102627.csv')
df = df.dropna(subset=['appearances'])           # drop players with no stats
df['appearances'] = pd.to_numeric(df['appearances'], errors='coerce')
df = df[df['appearances'] > 0].copy()

numeric_cols = ['appearances','goals','assists','yellow_cards','red_cards','clean_sheets','saves']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"Dataset: {len(df)} players across {df['team_name'].nunique()} teams")
print(df[['player_name','team_name','player_position','appearances','goals','assists']].to_string(index=False))

# ── 2. Feature Engineering ───────────────────────────────────────────────────
df['goals_per_game']   = df['goals']   / df['appearances']
df['assists_per_game'] = df['assists'] / df['appearances']
df['goal_contributions_per_game'] = (df['goals'] + df['assists']) / df['appearances']
df['discipline_score'] = df['yellow_cards'] + df['red_cards'] * 3  # weighted discipline
df['involvement']      = df['appearances'] / df['appearances'].max()  # 0-1 regularity

pos_enc = LabelEncoder()
df['position_enc'] = pos_enc.fit_transform(df['player_position'].fillna('Unknown'))

# ── 3. Team-level aggregations ───────────────────────────────────────────────
team_stats = df.groupby('team_name').agg(
    total_goals   = ('goals',   'sum'),
    total_assists = ('assists', 'sum'),
    total_apps    = ('appearances', 'sum'),
    avg_discipline= ('discipline_score', 'mean'),
    squad_size    = ('player_id', 'count')
).reset_index()
team_stats['team_goals_per_app'] = team_stats['total_goals'] / team_stats['total_apps']

print("\n\n══ TEAM SUMMARY (2025 Season) ══")
print(team_stats.sort_values('total_goals', ascending=False).to_string(index=False))

# ── 4. Player performance score ──────────────────────────────────────────────
# Weighted composite: goals carry most weight, assists secondary, discipline subtracts
df['performance_score'] = (
    df['goals_per_game']   * 3.0 +
    df['assists_per_game'] * 1.5 +
    df['involvement']      * 1.0 -
    df['discipline_score'] * 0.2
)

print("\n\n══ TOP 10 PERFORMERS (2025 Season) ══")
top = df.nlargest(10, 'performance_score')[
    ['player_name','team_name','player_position','appearances','goals','assists','performance_score']
].round(3)
print(top.to_string(index=False))

# ── 5. Train ML model to predict goals ───────────────────────────────────────
features = ['appearances','assists','yellow_cards','red_cards','position_enc',
            'assists_per_game','involvement','discipline_score']
target   = 'goals'

X = df[features].values
y = df[target].values

# Use multiple models, pick best via LOO-CV (small dataset)
models = {
    'Random Forest':        RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
    'Gradient Boosting':    GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    'Ridge Regression':     Ridge(alpha=1.0),
}

loo = LeaveOneOut()
print("\n\n══ MODEL EVALUATION (Leave-One-Out CV) ══")
best_model, best_mae = None, 9999
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_absolute_error')
    mae = -scores.mean()
    print(f"  {name:25s}  MAE = {mae:.2f} goals")
    if mae < best_mae:
        best_mae, best_model_name, best_model = mae, name, model

print(f"\n  ✓ Best model: {best_model_name} (MAE = {best_mae:.2f})")

# Fit on full data
best_model.fit(X, y)

# ── 6. Feature importance ────────────────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    print("\n\n══ FEATURE IMPORTANCE ══")
    fi = sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in fi:
        bar = '█' * int(imp * 40)
        print(f"  {feat:30s} {bar} {imp:.3f}")

# ── 7. Next-Season Projections ───────────────────────────────────────────────
# Assume next season = 12 games (slight increase); scale per-game rates
NEXT_SEASON_GAMES = 12

proj = df.copy()
proj['proj_appearances'] = (proj['involvement'] * NEXT_SEASON_GAMES).clip(upper=NEXT_SEASON_GAMES).round(0).astype(int)
proj['proj_assists']     = (proj['assists_per_game'] * proj['proj_appearances']).round(1)
proj['proj_yellow']      = (proj['yellow_cards'] / proj['appearances'] * proj['proj_appearances']).round(1)
proj['proj_red']         = (proj['red_cards']    / proj['appearances'] * proj['proj_appearances']).round(1)
proj['proj_discipline']  = proj['proj_yellow'] + proj['proj_red'] * 3
proj['proj_involvement'] = proj['proj_appearances'] / proj['proj_appearances'].max()
proj['proj_assists_per_game'] = proj['assists_per_game']  # assume same rate

X_proj = proj[['proj_appearances','proj_assists','proj_yellow','proj_red',
                'position_enc','proj_assists_per_game','proj_involvement','proj_discipline']].values

proj['projected_goals'] = best_model.predict(X_proj).clip(min=0).round(1)
proj['projected_goal_contributions'] = proj['projected_goals'] + proj['proj_assists']

print(f"\n\n══ NEXT SEASON PROJECTIONS ({NEXT_SEASON_GAMES} games) ══")
proj_display = proj.nlargest(15, 'projected_goals')[
    ['player_name','team_name','player_position',
     'proj_appearances','projected_goals','proj_assists','projected_goal_contributions']
].round(1)
proj_display.columns = ['Player','Team','Position','Proj Apps','Proj Goals','Proj Assists','Proj G+A']
print(proj_display.to_string(index=False))

# Team projections
team_proj = proj.groupby('team_name').agg(
    proj_goals = ('projected_goals', 'sum'),
    proj_assists= ('proj_assists', 'sum'),
    proj_apps  = ('proj_appearances', 'sum')
).reset_index().round(1)
team_proj['proj_goals_per_game'] = (team_proj['proj_goals'] / team_proj['proj_apps']).round(3)
team_proj = team_proj.sort_values('proj_goals', ascending=False)

print("\n\n══ TEAM PROJECTIONS (Next Season) ══")
print(team_proj.to_string(index=False))

# ── 8. Risk Assessment ───────────────────────────────────────────────────────
print("\n\n══ DISCIPLINE RISK FLAGS ══")
risk = proj[proj['proj_discipline'] >= 2][['player_name','team_name','player_position','proj_yellow','proj_red','proj_discipline']]
if len(risk):
    risk.columns = ['Player','Team','Position','Proj Yellows','Proj Reds','Risk Score']
    print(risk.sort_values('Risk Score', ascending=False).to_string(index=False))
else:
    print("  No high-discipline-risk players flagged.")

print("\n\n══ BREAKOUT CANDIDATES (High rate, low appearances) ══")
breakout = proj[(proj['goals_per_game'] >= 0.3) & (proj['appearances'] <= 6)][
    ['player_name','team_name','player_position','appearances','goals','goals_per_game','projected_goals']
].round(2)
if len(breakout):
    print(breakout.to_string(index=False))
else:
    print("  No breakout candidates identified.")

print("\n\n✅ Model training complete.")
print(f"   Algorithm : {best_model_name}")
print(f"   Accuracy  : ±{best_mae:.1f} goals (MAE on Leave-One-Out CV)")
print(f"   Players   : {len(df)} | Teams : {df['team_name'].nunique()}")

# ── 9. Generate HTML Report ──────────────────────────────────────────────────
print("\n📄 Generating HTML report...")

def pos_badge(pos):
    cls = {'Forward':'pos-Forward','Midfielder':'pos-Midfielder',
           'Defender':'pos-Defender','Goalkeeper':'pos-Goalkeeper'}.get(pos, 'pos-Midfielder')
    short = {'Forward':'FWD','Midfielder':'MID','Defender':'DEF','Goalkeeper':'GK'}.get(pos, pos[:3].upper())
    return f'<span class="pos-badge {cls}">{short}</span>'

def bar_html(val, max_val, width=100):
    w = int((val / max_val) * width) if max_val > 0 else 0
    return f'<div class="bar-wrap"><div class="bar" style="width:{w}px"></div> {val:.3f}</div>'

# Build top performers rows
top10_rows = ""
medals = {1:'medal-1', 2:'medal-2', 3:'medal-3'}
perf_sorted = df.nlargest(10, 'performance_score').reset_index(drop=True)
for i, row in perf_sorted.iterrows():
    rank = i + 1
    medal = medals.get(rank, '')
    gc = int(row['goals'] + row['assists'])
    top10_rows += f"""<tr>
      <td class="rank {medal}">{rank}</td>
      <td>{row['player_name']}</td><td>{row['team_name']}</td>
      <td>{pos_badge(row['player_position'])}</td>
      <td>{int(row['appearances'])}</td><td>{int(row['goals'])}</td>
      <td>{int(row['assists'])}</td><td>{gc}</td>
      <td>{row['performance_score']:.2f}</td>
    </tr>"""

# Build team summary rows
team_sorted = team_stats.sort_values('total_goals', ascending=False).reset_index(drop=True)
max_gpg = team_sorted['team_goals_per_app'].max()
team_rows = ""
for i, row in team_sorted.iterrows():
    rank = i + 1
    medal = medals.get(rank, '')
    team_rows += f"""<tr>
      <td class="rank {medal}">{rank}</td>
      <td>{row['team_name']}</td>
      <td>{int(row['total_goals'])}</td><td>{int(row['total_assists'])}</td>
      <td>{int(row['total_apps'])}</td>
      <td>{bar_html(row['team_goals_per_app'], max_gpg)}</td>
      <td>{int(row['squad_size'])}</td>
    </tr>"""

# Build projections rows
proj_sorted = proj.nlargest(10, 'projected_goals').reset_index(drop=True)
proj_rows = ""
for i, row in proj_sorted.iterrows():
    rank = i + 1
    medal = medals.get(rank, '')
    proj_rows += f"""<tr>
      <td class="rank {medal}">{rank}</td>
      <td>{row['player_name']}</td><td>{row['team_name']}</td>
      <td>{pos_badge(row['player_position'])}</td>
      <td>{int(row['proj_appearances'])}</td>
      <td><strong>{row['projected_goals']:.1f}</strong></td>
      <td>{row['proj_assists']:.1f}</td>
      <td>{row['projected_goal_contributions']:.1f}</td>
    </tr>"""

# Build team projection rows
team_proj_sorted = team_proj.sort_values('proj_goals', ascending=False).reset_index(drop=True)
# compute 2025 totals for trend
goals_2025 = team_stats.set_index('team_name')['total_goals'].to_dict()
team_proj_rows = ""
for i, row in team_proj_sorted.iterrows():
    rank = i + 1
    medal = medals.get(rank, '')
    g25 = goals_2025.get(row['team_name'], 0)
    pct = ((row['proj_goals'] - g25) / g25 * 100) if g25 > 0 else 0
    arrow = "▲" if pct >= 0 else "▼"
    css = "risk-low" if pct >= 0 else "risk-med"
    team_proj_rows += f"""<tr>
      <td class="rank {medal}">{rank}</td>
      <td>{row['team_name']}</td>
      <td>{row['proj_goals']:.1f}</td><td>{row['proj_assists']:.1f}</td>
      <td>{row['proj_goals_per_game']:.3f}</td>
      <td class="{css}">{arrow} {abs(pct):.0f}% goals</td>
    </tr>"""

# Build risk rows
risk_df = proj[proj['proj_discipline'] >= 2].sort_values('proj_discipline', ascending=False)
risk_rows = ""
for _, row in risk_df.iterrows():
    css = "risk-high" if row['proj_discipline'] >= 4 else "risk-med"
    level = "HIGH" if row['proj_discipline'] >= 4 else "MED"
    risk_rows += f"""<tr>
      <td>{row['player_name']}</td><td>{row['team_name']}</td>
      <td>{row['proj_yellow']:.1f}</td><td>{row['proj_red']:.1f}</td>
      <td><span class="{css}">● {level} ({row['proj_discipline']:.1f})</span></td>
    </tr>"""
if not risk_rows:
    risk_rows = '<tr><td colspan="5" style="color:#8b949e;text-align:center">No high-risk players</td></tr>'

# Build breakout rows
breakout_df = proj[(proj['goals_per_game'] >= 0.3) & (proj['appearances'] <= 6)]
breakout_rows = ""
for i, (_, row) in enumerate(breakout_df.iterrows()):
    tag = ' <span class="breakout-chip">🔥 Hottest</span>' if i == 0 else ''
    breakout_rows += f"""<tr>
      <td>{row['player_name']}{tag}</td><td>{row['team_name']}</td>
      <td>{int(row['appearances'])}</td>
      <td><strong class="risk-low">{row['goals_per_game']:.2f}</strong></td>
      <td>{row['projected_goals']:.1f}</td>
    </tr>"""

# Feature importance section
fi_rows = ""
if hasattr(best_model, 'feature_importances_'):
    fi_pairs = sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in fi_pairs:
        pct = int(imp * 100)
        fi_rows += f"""<div class="feat-row">
          <div class="feat-name">{feat.replace('_',' ').title()}</div>
          <div class="feat-bar"><div class="feat-fill" style="width:{pct}%"></div></div>
          <div class="feat-val">{imp:.3f}</div>
        </div>"""
else:
    # Ridge doesn't have feature_importances_, show coef magnitudes
    coefs = np.abs(best_model.coef_)
    coef_max = coefs.max()
    fi_pairs = sorted(zip(features, coefs), key=lambda x: -x[1])
    for feat, coef in fi_pairs:
        pct = int((coef / coef_max) * 100)
        fi_rows += f"""<div class="feat-row">
          <div class="feat-name">{feat.replace('_',' ').title()}</div>
          <div class="feat-bar"><div class="feat-fill" style="width:{pct}%"></div></div>
          <div class="feat-val">{coef:.3f}</div>
        </div>"""

total_goals = int(df['goals'].sum())
total_assists = int(df['assists'].sum())
num_players = len(df)
num_teams = df['team_name'].nunique()

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Football ML Predictions Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; }}
  .hero {{ background: linear-gradient(135deg, #1a1f35 0%, #0d1117 100%);
           border-bottom: 1px solid #30363d; padding: 40px 32px 32px; }}
  .hero h1 {{ font-size: 2rem; font-weight: 700; color: #58a6ff; }}
  .hero p  {{ color: #8b949e; margin-top: 6px; font-size: 0.95rem; }}
  .badge   {{ display:inline-block; background:#1f6feb22; border:1px solid #1f6feb55;
              color:#58a6ff; border-radius:20px; padding:3px 12px; font-size:0.8rem; margin-top:10px; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 32px; }}
  section {{ margin-bottom: 40px; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; color: #f0f6fc; border-left: 3px solid #58a6ff;
        padding-left: 12px; margin-bottom: 16px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; }}
  .card .val {{ font-size: 2rem; font-weight: 700; color: #58a6ff; }}
  .card .lbl {{ font-size: 0.8rem; color: #8b949e; margin-top: 4px; }}
  .card .sub {{ font-size: 0.75rem; color: #6e7681; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
  thead th {{ background: #1c2128; color: #8b949e; font-weight: 500; text-align: left;
              padding: 10px 14px; border-bottom: 1px solid #30363d; font-size: 0.78rem;
              text-transform: uppercase; letter-spacing: 0.04em; }}
  tbody tr {{ border-bottom: 1px solid #21262d; transition: background 0.15s; }}
  tbody tr:hover {{ background: #1c2128; }}
  tbody td {{ padding: 10px 14px; color: #c9d1d9; }}
  .tbl-wrap {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; overflow: hidden; }}
  .rank {{ font-weight: 700; color: #58a6ff; width: 30px; }}
  .medal-1 {{ color: #ffd700; }}
  .medal-2 {{ color: #c0c0c0; }}
  .medal-3 {{ color: #cd7f32; }}
  .pos-badge {{ display:inline-block; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }}
  .pos-Forward    {{ background:#1a3a1a; color:#56d364; }}
  .pos-Midfielder {{ background:#1a2a3a; color:#58a6ff; }}
  .pos-Defender   {{ background:#2a1a3a; color:#bc8cff; }}
  .pos-Goalkeeper {{ background:#3a2a1a; color:#e3b341; }}
  .bar-wrap {{ display:flex; align-items:center; gap:10px; }}
  .bar {{ height:8px; border-radius:4px; background:linear-gradient(90deg,#1f6feb,#58a6ff); min-width:2px; }}
  .risk-high {{ color: #f85149; }}
  .risk-med  {{ color: #e3b341; }}
  .risk-low  {{ color: #56d364; }}
  .breakout-chip {{ background:#1a3a1a; color:#56d364; border:1px solid #56d36444;
                    border-radius:12px; padding:2px 10px; font-size:0.72rem; font-weight:600; }}
  .model-box {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px; }}
  .model-row {{ display:flex; justify-content:space-between; align-items:center;
                padding:10px 0; border-bottom:1px solid #21262d; }}
  .model-row:last-child {{ border-bottom:none; }}
  .model-name {{ color:#c9d1d9; font-size:0.9rem; }}
  .model-mae {{ font-size:0.85rem; }}
  .model-best {{ color:#56d364; font-weight:600; }}
  .model-other {{ color:#8b949e; }}
  .winner-tag {{ background:#1a3a1a; color:#56d364; border-radius:6px; padding:1px 8px; font-size:0.72rem; }}
  .feat-row {{ display:flex; align-items:center; gap:12px; padding:6px 0; }}
  .feat-name {{ width:180px; color:#c9d1d9; font-size:0.85rem; }}
  .feat-bar  {{ flex:1; height:8px; border-radius:4px; background:#21262d; }}
  .feat-fill {{ height:100%; border-radius:4px; background:linear-gradient(90deg,#58a6ff,#a78bfa); }}
  .feat-val  {{ width:48px; text-align:right; color:#8b949e; font-size:0.8rem; }}
  .note {{ background:#1c2128; border:1px solid #30363d; border-radius:8px;
           padding:14px 18px; font-size:0.82rem; color:#8b949e; margin-top:20px; line-height:1.6; }}
  .note strong {{ color:#e3b341; }}
  footer {{ text-align:center; padding:32px; color:#6e7681; font-size:0.78rem; border-top:1px solid #21262d; }}
</style>
</head>
<body>
<div class="hero">
  <h1>⚽ Football ML Predictions</h1>
  <p>Machine learning model trained on 2025 season data — projecting next season performance</p>
  <span class="badge">{best_model_name} · LOO-CV Validated · ±{best_mae:.1f} goal accuracy</span>
</div>
<main>
  <section>
    <h2>2025 Season Overview</h2>
    <div class="grid-4">
      <div class="card"><div class="val">{num_players}</div><div class="lbl">Players analysed</div><div class="sub">{num_teams} teams · 1 season</div></div>
      <div class="card"><div class="val">{total_goals}</div><div class="lbl">Total Goals Scored</div><div class="sub">across all teams</div></div>
      <div class="card"><div class="val">{total_assists}</div><div class="lbl">Total Assists</div><div class="sub">across all teams</div></div>
      <div class="card"><div class="val">±{best_mae:.1f}</div><div class="lbl">Model Accuracy</div><div class="sub">mean absolute error (goals)</div></div>
    </div>
  </section>
  <section>
    <h2>Team Performance — 2025 Season</h2>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>#</th><th>Team</th><th>Goals</th><th>Assists</th><th>Apps</th><th>Goals / App</th><th>Squad</th></tr></thead>
        <tbody>{team_rows}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Top 10 Performers — 2025 Season</h2>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>#</th><th>Player</th><th>Team</th><th>Position</th><th>Apps</th><th>Goals</th><th>Assists</th><th>G+A</th><th>Score</th></tr></thead>
        <tbody>{top10_rows}</tbody>
      </table>
    </div>
  </section>
  <section>
    <div class="grid-2">
      <div>
        <h2>Model Comparison</h2>
        <div class="model-box">
          <div class="model-row">
            <span class="model-name">🏆 {best_model_name} <span class="winner-tag">SELECTED</span></span>
            <span class="model-mae model-best">MAE = ±{best_mae:.2f} goals</span>
          </div>
          {''.join(f'<div class="model-row"><span class="model-name">{n}</span><span class="model-mae model-other">MAE = ±{-cross_val_score(m, X, y, cv=loo, scoring="neg_mean_absolute_error").mean():.2f} goals</span></div>' for n, m in models.items() if n != best_model_name)}
        </div>
        <div class="note"><strong>Note:</strong> Accuracy validated using Leave-One-Out Cross-Validation — the most reliable method for small datasets.</div>
      </div>
      <div>
        <h2>Feature Importance</h2>
        <div class="model-box">{fi_rows}</div>
      </div>
    </div>
  </section>
  <section>
    <h2>Next Season Projections (12 Games)</h2>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>#</th><th>Player</th><th>Team</th><th>Position</th><th>Proj Apps</th><th>Proj Goals</th><th>Proj Assists</th><th>Proj G+A</th></tr></thead>
        <tbody>{proj_rows}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Team Projections — Next Season</h2>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>#</th><th>Team</th><th>Proj Goals</th><th>Proj Assists</th><th>Goals / App</th><th>Trend vs 2025</th></tr></thead>
        <tbody>{team_proj_rows}</tbody>
      </table>
    </div>
  </section>
  <section>
    <div class="grid-2">
      <div>
        <h2>⚠️ Discipline Risk Flags</h2>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>Player</th><th>Team</th><th>Proj Yellows</th><th>Proj Reds</th><th>Risk</th></tr></thead>
            <tbody>{risk_rows}</tbody>
          </table>
        </div>
      </div>
      <div>
        <h2>🚀 Breakout Candidates</h2>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>Player</th><th>Team</th><th>Apps</th><th>Goals/Game</th><th>Proj Goals</th></tr></thead>
            <tbody>{breakout_rows}</tbody>
          </table>
        </div>
      </div>
    </div>
  </section>
</main>
<footer>Generated by Football ML Predictor · {best_model_name} · 2025 Season Data · {num_players} Players · {num_teams} Teams</footer>
</body>
</html>"""

output_path = "footy_ml_report.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"   ✅ HTML report saved to: {output_path}")
