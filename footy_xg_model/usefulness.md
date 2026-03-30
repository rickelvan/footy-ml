# How to Explain and Use the xG Model Results

This guide shows how to explain the FootyStats Expected Goals (xG) model results to different people and how useful these results can be in practice.

---

## 1. Explaining Results to a General Person (Simple Version)

### What is xG?

**In one sentence:**  
xG is the probability that a shot becomes a goal — a number between 0 and 1.

| xG value | In plain words |
|----------|----------------|
| **0.05** | Very unlikely — e.g. long shot from outside the box |
| **0.20** | Possible — decent chance but not easy |
| **0.50** | 50/50 — a typical one-on-one or clear cut chance |
| **0.75** | Very likely — tap-in or close-range finish |
| **0.90** | Almost certain — open goal or point-blank header |

### What does the model actually do?

The model looks at each shot and asks: *"Of all similar shots in the past, how many went in?"*  
It uses things like:

- How far from goal
- How central the shot is (angle)
- Whether it was a header or a shot with the foot
- Match situation (minute, score, pressure)

It then turns that into a probability. For example, **xG = 0.31** means the model estimates a 31% chance that shot would be a goal.

### How to explain example outputs

| Output example | How to explain it |
|----------------|-------------------|
| Shot A → xG = 0.31 | "That shot had about a 1-in-3 chance of going in based on similar chances." |
| Shot B → xG = 0.08 | "A difficult chance — only about 8% of shots like that normally score." |
| Shot C → xG = 0.62 | "A good chance — more than 60% of similar shots result in a goal." |
| Team total xG = 2.4 | "On average, a team with those chances would score about 2 to 3 goals." |
| xG 2.4 but only 1 goal | "They created good chances but didn’t finish well, or were unlucky." |

---

## 2. How Useful Are These Results?

### Real-world value

| Area | Usefulness | Example |
|------|------------|---------|
| **Match verdict** | High | "We had 2.1 xG vs their 0.8 — we created better chances and arguably deserved more." |
| **Player evaluation** | High | Compare xG vs actual goals to see who finishes well and who wastes chances. |
| **Recruitment** | Medium–High | Separate chance quality from finishing skill when comparing strikers. |
| **Tactical analysis** | Medium–High | Identify where and how to create higher-quality chances. |
| **Training focus** | Medium | Use xG to prioritise finishing work on high-value situations. |
| **Broadcast and media** | Medium | "That shot had 0.65 xG — a big chance that usually goes in." |
| **Fan engagement** | Medium | Give fans a clear metric to talk about chance quality. |

### Main benefits

1. **Standardises chance quality** — A tap-in and a 30-yard shot are not the same; xG quantifies that.
2. **Separates skill from luck** — If a player consistently scores above xG, they are finishing well; below xG suggests waste or bad luck.
3. **Improves decisions** — Coaches can focus on creating high-xG chances and improving conversion.
4. **Clear communication** — "2.5 xG" is easier to interpret than "lots of chances."
5. **Measurable and reproducible** — Results come from a documented model, not gut feeling alone.

---

## 3. Explaining Results to Different Audiences

### Coaches and staff

**Focus on:** Decisions, player roles, and what to change.

| Point | What to say |
|-------|-------------|
| **xG vs goals** | "We created 2.1 xG but scored once — we’re not finishing as well as our chances suggest." |
| **Shot quality** | "Many shots were low xG (e.g. &lt;0.10). We need to work on creating higher-quality chances in the box." |
| **Player selection** | "Player A converts above xG; Player B underperforms. Worth looking at composure and decision-making for Player B." |
| **Tactics** | "Our xG from inside the box is low — we should improve ball progression and movement in the final third." |

**Avoid:** Log loss, ROC-AUC, Brier score, hyperparameters.  
**Use:** Simple visuals (shot maps, xG bars), team/player xG totals, xG vs goals over time.

---

### Fans

**Focus on:** "Did we deserve it?" and "Who played well?"

| Point | What to say |
|-------|-------------|
| **Match verdict** | "The xG was 2.1–0.8 in our favour — we created better chances and deserved more than a draw." |
| **Luck** | "Their keeper made several saves on high-xG shots; we were a bit unlucky." |
| **Player performance** | "Our striker had 0.9 xG — on average that would be almost one goal, so a good outing." |

**Keep it short:**
- xG ≈ "how many goals we *should* have scored"
- Actual goals vs xG ≈ "luck vs conversion"

---

### Technical people (analysts, data scientists)

**Focus on:** Methodology, metrics, and how to interpret or improve the model.

| Topic | What to cover |
|-------|---------------|
| **Data** | Synthetic shot-level data derived from player-season stats; assumptions in `config.py`. |
| **Features** | Distance, angle, body part, shot type, position, match minute, score difference, pressure. |
| **Model choice** | Best of Logistic Regression / Random Forest / Gradient Boosting by validation log loss, then calibrated. |
| **Metrics** | Log loss, ROC-AUC, Brier score, precision/recall at 0.5 threshold. |
| **Calibration** | Isotonic regression so predicted probabilities align better with observed outcomes. |

Share the full HTML evaluation report, feature importance plots, and confusion matrix.

---

## 4. Reading the HTML Report — Step-by-Step

When you open `xg_evaluation_report.html`, here’s what each part means:

| Section | What it shows | Plain-language takeaway |
|---------|---------------|-------------------------|
| **Dataset Overview** | Total shots, train/test split, goal rate | How much data the model used and how often shots went in. |
| **Test Set Metrics** | Log loss, ROC-AUC, Brier score, precision, recall | How well the model predicts; lower log loss and Brier score, higher ROC-AUC = better. |
| **Model Comparison** | Which model was chosen (e.g. Gradient Boosting) | Which algorithm performed best before calibration. |
| **Best Model** | Hyperparameters | Settings used for the chosen model. |
| **ROC Curve** | How well the model separates goals from non-goals | Steeper curve = better discrimination. |
| **Precision–Recall Curve** | Trade-off between precision and recall | Shows how the model balances correct predictions. |
| **Confusion Matrix** | True/false positives and negatives at 0.5 threshold | How many goals and non-goals the model got right or wrong. |
| **Calibration Curve** | Whether predicted probabilities match reality | Points near the diagonal = well-calibrated xG. |
| **Feature Importance** | Which factors matter most (e.g. distance, angle) | Distance and angle usually drive xG most. |
| **Example xG Predictions** | Sample shots with player, location, xG | Concrete examples of how the model scores shots. |

---

## 5. Practical Uses in Detail

| Use case | How xG helps | Who benefits |
|----------|--------------|--------------|
| **Match analysis** | Compare xG by team and by half; see if a result was fair or lucky | Coaches, analysts, fans |
| **Player evaluation** | xG vs actual goals per player to find over- and underperformers | Coaches, recruitment |
| **Broadcast / storytelling** | Show xG per shot during replays; "That shot had 0.65 xG." | Commentators, media |
| **Recruitment** | Separate finishing skill from chance quality when comparing strikers | Scouts, sporting directors |
| **Strategy & tactics** | Find which areas and shot types yield high xG; adjust plans | Coaches, analysts |
| **Set pieces** | Use xG from corners and free kicks to judge set-piece effectiveness | Set-piece coaches |
| **Fan engagement** | Give fans a clear metric to discuss chance quality and "deserved" results | Clubs, media |

---

## 6. Limits and Caveats

| Limitation | Explanation |
|------------|-------------|
| **Synthetic training data** | Shots are simulated from player-season stats, not real event data. Results may differ from models trained on full tracking/event data. |
| **Limited context** | No defender/goalkeeper positions, pressure intensity, or passing sequences. Affects accuracy in marginal situations. |
| **Historical bias** | The model reflects the shot distribution in the training data. Different leagues or styles may not match. |
| **Single snapshot** | Each shot is evaluated at the moment of the strike; no info on what happened before or after. |
| **Probability, not fate** | xG is a long-run average. A 0.90 xG shot can miss; a 0.05 xG shot can go in. |
| **Calibration scope** | Calibration is on the validation set. Performance may drift on new seasons or competitions. |

**Best practice:** Treat xG as a *tool for insight*, not a replacement for watching games or expert judgement.

---

## 7. Suggested Workflow for Presenting Results

1. **Know your audience** — Coaches want tactics and actions; fans want "did we deserve it?"; technical people want metrics and methodology.
2. **Start with the headline** — e.g. "We created 2.1 xG but scored 1" or "Striker X is outperforming his xG by 20%."
3. **Use visuals first** — Shot maps, xG bars, simple comparison tables before text or tables.
4. **Explain in plain language** — Define xG as "probability of scoring that shot based on similar shots in the past."
5. **Add context** — Sample size, time period, competition, and whether data is synthetic.
6. **Call out limitations** — Brief mention of data type and assumptions so no one over-interprets.
7. **Make it actionable (for coaches)** — Tie insights to decisions: line-up, training focus, tactical changes.
8. **Have the full report ready** — For technical stakeholders, share the HTML report and links to code/config.

---

## 8. Summary — Why These Results Matter

- **xG standardises chance quality** — A tap-in and a 30-yard strike are not the same; xG quantifies that difference.
- **It separates skill from luck** — Consistent over- or underperformance vs xG highlights finishing ability or lack of it.
- **It supports better decisions** — Coaches and analysts can prioritise creating high-xG chances and improving conversion.
- **It makes storytelling clearer** — "We had 2.5 xG" is easier to understand than "we had lots of chances."
- **It is measurable and reproducible** — Unlike purely subjective opinions, xG comes from a documented model and metrics.

When used with its caveats in mind, xG is a practical tool for understanding finishing, chance creation, and performance, and for communicating that understanding across different audiences.
