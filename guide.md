Advanced Machine Learning Enhancements for FootyStats:
A Data-Driven Approach for Amateur Football Analytics
Katende Derrick Elvan Uganda Christian University
2026

Contents
 
1	Introduction	3
2	Dynamic Team Strength and Scoreline Prediction Model	4
2.1	Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .	4
2.2	Why This Model is Important . . . . . . . . . . . . . . . . . . . . . . . .	4
2.3	Model Formulation  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .	4
2.4	Visualizations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .	4
2.5	Implementation Prompt	. . . . . . . . . . . . . . . . . . . . . . . . . . .	4
3	Bayesian Player Ability Model	6
3.1	Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .	6
3.2	Why This Model is Important . . . . . . . . . . . . . . . . . . . . . . . .	6
3.3	Model Idea	. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .	6
3.4	Visualizations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .	6
3.5	Implementation Prompt	. . . . . . . . . . . . . . . . . . . . . . . . . . .	6
4	Player Archetype Clustering	8
4.1	Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .	8
4.2	Why This Model is Important . . . . . . . . . . . . . . . . . . . . . . . .	8
4.3	Method	. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .	8
4.4	Visualizations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .	8
4.5	Implementation Prompt	. . . . . . . . . . . . . . . . . . . . . . . . . . .	8
5	Event Sequence Prediction Model (Optional)	10
5.1	Overview	10
5.2	Why This Model is Important	10
5.3	Approach	10
5.4	Visualizations	10
5.5	Implementation Prompt	10
6	Conclusion	11

 
1	Introduction
This project enhances the FootyStats platform by introducing advanced machine learn- ing models specifically designed for amateur football environments. Unlike professional leagues, amateur football lacks rich datasets such as player tracking, shot coordinates, and detailed event streams. Therefore, this work focuses on models that operate effectively using limited but reliable data such as goals, assists, cards, and match outcomes.
The goal is to demonstrate that meaningful football intelligence can still be derived from sparse datasets through appropriate model selection, probabilistic reasoning, and interpretable outputs.
 
2	Dynamic Team Strength and Scoreline Prediction Model
2.1	Overview
This model estimates match outcomes and exact scorelines using a probabilistic frame- work based on team attack and defense strengths. It models the number of goals scored by each team using a Poisson distribution.

2.2	Why This Model is Important
•	Works with minimal data (goals only)
•	Produces full probability distributions instead of single predictions
•	Widely used in football research (Dixon-Coles model)
•	Captures team strength evolution over time

2.3	Model Formulation
Goalshome ∼ Poisson(λhome) Goalsaway ∼ Poisson(λaway)
Where:
λ = AttackStrength × DefenseWeakness × HomeAdvantage

2.4	Visualizations
•	Scoreline probability heatmap (0-0 to 5-5)
•	Team strength over time graph
•	Home/Draw/Away probability bar chart

2.5	Implementation Prompt
Prompt:
Build a football match prediction model using a Poisson distribution. Inputs:
-	Historical match results (home team, away team, goals scored)
-	Features:  rolling  goals  scored,  goals  conceded,  home  advantage
Tasks:
1.	Compute attack and defense strength for each team
2.	Estimate expected goals (lambda values)
3.	Generate probability matrix for scorelines (0-0 to 5-5)
 
4.	Calculate  home/draw/away  probabilities
5.	Plot:
-	Heatmap of scoreline probabilities
-	Line graph of team strength over time
-	Bar chart for match outcome probabilities
Use Python (pandas, numpy, matplotlib, seaborn)
Ensure output is visually clear and labeled for presentation
 
3	Bayesian Player Ability Model
3.1	Overview
This model estimates the true underlying ability of players using goals, assists, and cards while accounting for differences in appearances.

3.2	Why This Model is Important
•	Adjusts for players with fewer matches
•	Avoids misleading raw statistics
•	Introduces uncertainty estimation
•	Uses proper statistical modeling

3.3	Model Idea
Goalsi ∼ Poisson(λi × Appearancesi)
Where λi represents player ability.

3.4	Visualizations
•	Player ability ranking with confidence intervals
•	Scatter plot: Goals vs Assists
•	Radar chart for player profiles

3.5	Implementation Prompt
Prompt:
Build a Bayesian model to estimate player ability using goals, assists, and cards. Inputs:
- Player statistics: goals, assists, cards, appearances
Tasks:
1.	Model goals and assists as Poisson processes
2.	Adjust for appearances (per-match normalization)
3.	Estimate player ability scores
4.	Compute uncertainty intervals
Visualizations:
-	Bar chart of top players with error bars
-	Scatter plot (goals vs assists)
-	Radar chart showing player profile
 
Use Python (PyMC or simple Bayesian approximation) Ensure results are interpretable and visually clean
 
4	Player Archetype Clustering
4.1	Overview
This model groups players into archetypes based on their statistical profiles using unsu- pervised learning.

4.2	Why This Model is Important
•	Discovers hidden patterns in player behavior
•	Helps identify roles (finisher, creator, aggressive player)
•	Works without labeled data

4.3	Method
•	Normalize features per appearance
•	Reduce dimensions using UMAP
•	Cluster using HDBSCAN or K-Means

4.4	Visualizations
•	2D player clustering map
•	Cluster-based player grouping
•	Archetype summaries

4.5	Implementation Prompt
Prompt:
Perform clustering on football players using their statistics. Inputs:
-	goals per appearance
-	assists  per  appearance
-	cards per appearance
Tasks:
1.	Normalize data
2.	Apply  dimensionality  reduction  (UMAP)
3.	Cluster players using HDBSCAN or KMeans
4.	Label clusters (e.g., finisher, playmaker)
Visualizations:
-	2D scatter plot of players
-	Cluster coloring
 
-	Highlight example players per cluster
Use Python (sklearn, umap-learn, matplotlib) Make visualization presentation-ready
 
5	Event Sequence Prediction Model (Optional)
5.1	Overview
This model predicts the next event in a match using sequence modeling techniques.

5.2	Why This Model is Important
•	Introduces temporal modeling
•	Adds deep learning component
•	Captures match flow patterns

5.3	Approach
•	Use event sequences (GOAL, CARD, etc.)
•	Build baseline Markov model
•	Improve using GRU neural network

5.4	Visualizations
•	Transition matrix heatmap
•	Training loss curve
•	Prediction accuracy chart

5.5	Implementation Prompt
Prompt:
Build a sequence model to predict the next football match event.
Inputs:
- Ordered match events (goal, yellow card, red card)
Tasks:
1.	Encode events as sequences
2.	Build baseline Markov transition model
3.	Build GRU neural network model
4.	Compare performance
Visualizations:
-	Transition  matrix  heatmap
-	Confusion matrix
-	Training/validation loss curves
Use Python (TensorFlow or PyTorch) Ensure model comparison is clear
 
6	Conclusion
This work demonstrates that advanced machine learning techniques can be effectively applied to amateur football data. By focusing on probabilistic modeling, Bayesian infer- ence, clustering, and sequence learning, the system achieves a level of analytical depth comparable to professional football analytics while remaining realistic within data con- straints.
These models significantly enhance the intelligence layer of FootyStats and provide meaningful insights into team performance, player ability, and match dynamics.
