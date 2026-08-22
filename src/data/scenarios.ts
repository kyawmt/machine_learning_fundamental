import type { Scenario } from "../components/ScenarioCard";

export const SCENARIOS: Scenario[] = [
  {
    id: "s1",
    title: "The fraud classifier that looks great",
    setup:
      "A team ships a fraud detection model. In their report: **99% accuracy** and **60% recall**. The base fraud rate is 1%.",
    stats: [
      { label: "Accuracy", value: "99%", tone: "good" },
      { label: "Recall", value: "60%", tone: "warn" },
      { label: "Base rate", value: "1%", tone: "neutral" },
    ],
    question: "Is this a good model?",
    answer:
      "The 99% accuracy tells you nothing — predicting 'legitimate' for everything scores exactly 99% here. So the only informative number in that report is the recall, and even that is half the picture.\n\n60% recall means four out of every ten fraudulent transactions get through. Whether that's good depends entirely on the **precision**, which they haven't reported. At 60% recall with 80% precision, you're catching most fraud with a manageable alert queue — that's a decent model. At 60% recall with 5% precision, your analysts are wading through nineteen false alarms for every real case, and the system is probably unusable in practice.\n\nThe deeper point: accuracy was never the right metric here, and quoting it suggests the team hasn't thought about the imbalance.",
    nextSteps:
      "- Ask for the full confusion matrix, and precision alongside recall.\n- Ask for PR-AUC rather than ROC-AUC, given the 1% base rate.\n- Ask what happens operationally when the model fires — auto-decline, or human review? That determines which error is expensive.\n- Ask about the alert volume relative to review capacity; precision@K may be the metric that actually matters.\n- Quantify the two error costs in currency, then choose the threshold that minimises expected cost.",
  },
  {
    id: "s2",
    title: "The 24-point gap",
    setup: "A gradient boosting model on 12,000 rows of tabular data.",
    stats: [
      { label: "Train accuracy", value: "99%", tone: "warn" },
      { label: "Validation accuracy", value: "75%", tone: "bad" },
    ],
    question: "What is happening, and what would you try?",
    answer:
      "Classic **overfitting** — high variance. A 24-point gap means the model has enough capacity to memorise the training rows and is fitting sample-specific noise rather than the generalisable pattern.\n\nBefore reaching for fixes, I'd rule out an infrastructure cause: is the validation set drawn from the same distribution, does the split respect any time or group structure, and is the validation set large enough for 75% to be a stable estimate.",
    nextSteps:
      "Assuming the split is sound, in order of expected payoff:\n- **Constrain the model** — lower max_depth, raise min_child_weight, lower num_leaves.\n- **Add regularization** — reg_lambda / reg_alpha, and subsample plus colsample_bytree to add randomness.\n- **Early stopping** on the validation set; the model is almost certainly running too many rounds.\n- **Lower the learning rate** and let early stopping choose the tree count.\n- **More data**, if it's obtainable — the most reliable fix for variance.\n- **Fewer features**, if the feature count is large relative to 12,000 rows.",
  },
  {
    id: "s3",
    title: "Both numbers are bad",
    setup: "A logistic regression on a customer conversion problem.",
    stats: [
      { label: "Train accuracy", value: "65%", tone: "bad" },
      { label: "Validation accuracy", value: "63%", tone: "bad" },
    ],
    question: "What is happening?",
    answer:
      "**Underfitting** — high bias. The two scores being low *and* close is the signature: the model isn't even fitting the data it can see, so generalisation isn't the bottleneck. The model, or the features, cannot express the pattern.\n\nOne thing to check first: what's the majority-class baseline? If 62% of customers don't convert, then 63% is essentially no signal at all, and that reframes the problem — it may be that the features genuinely don't carry information about the target.",
    nextSteps:
      "- **Better features first** — on tabular data this is usually the biggest lever. Interactions, ratios, aggregates, time-since-last-event.\n- **More capacity** — move from logistic regression to a gradient boosting model, which captures interactions natively.\n- **Reduce regularization** — increase C in scikit-learn's logistic regression.\n- **Check convergence** — did the solver actually converge, or hit max_iter?\n- **Sanity-check the data** — a misaligned join or a scrambled target looks exactly like underfitting.",
  },
  {
    id: "s4",
    title: "The suspicious improvement",
    setup:
      "You add a batch of new features to a churn model. Test AUC jumps from **0.78 to 0.96** in one iteration. The team wants to ship it today.",
    stats: [
      { label: "AUC before", value: "0.78", tone: "neutral" },
      { label: "AUC after", value: "0.96", tone: "warn" },
    ],
    question: "What should you investigate before shipping?",
    answer:
      "**Data leakage**, almost certainly. A jump that size from a feature batch is not a modelling breakthrough — on a real churn problem it means one of the new features carries information about the outcome that won't exist at prediction time.\n\nThe usual culprits: a feature computed over a window that extends past the prediction timestamp, a field populated by a downstream process that only runs after the customer cancels (a cancellation reason code, a final invoice, a support ticket category), or an aggregate built from a snapshot rather than as-of the event.",
    nextSteps:
      "- Look at **feature importance** and identify the top new feature.\n- For that feature, trace it to its source and establish **when its value is written** relative to the prediction moment.\n- Run an **ablation**: drop it and retrain. If AUC collapses back to ~0.78, it's doing all the work on its own — a red flag in itself.\n- **Group the target by that feature** — leakage is usually visually obvious.\n- Validate on a **strictly later time period**. Leakage rarely survives a temporal holdout.",
  },
  {
    id: "s5",
    title: "Medical screening",
    setup:
      "You're building a first-line screening model for a serious but treatable disease. Positive cases are referred for a confirmatory test that is accurate but costly and mildly invasive.",
    question: "Precision or recall — which matters more, and how would you decide?",
    answer:
      "**Recall**, in this setup — but the reasoning matters more than the answer, and I'd say the reasoning out loud.\n\nA false negative means a patient with the disease is told they're clear and goes home untreated. A false positive means a patient undergoes an unnecessary confirmatory test. In a screening context those costs are wildly asymmetric, so you accept a lot of false positives to avoid a miss. That's why real screening programmes are explicitly designed as high-sensitivity first stages backed by a specific second stage.\n\nBut it isn't unconditional. The tradeoff shifts if the confirmatory test is itself risky, if false positives cause serious psychological harm, or if the referral capacity is limited — at some point a flood of referrals means the genuinely sick wait longer, and pushing recall further makes outcomes worse, not better.",
    nextSteps:
      "- Frame it as expected harm rather than a metric: what is the cost of a missed case versus an unnecessary follow-up?\n- Pick a target recall with clinicians (screening programmes often specify sensitivity, e.g. ≥ 95%), then maximise precision at that recall.\n- Report the operating point explicitly, and check calibration, since clinical decisions use the probability.\n- Slice performance by demographic subgroup — an aggregate recall of 95% hiding 70% for one group is a serious problem.",
  },
  {
    id: "s6",
    title: "CV said 0.88, production says 0.61",
    setup:
      "A model with a stable 5-fold CV AUC of 0.88 (± 0.01) is deployed. After a month of live data, measured AUC is 0.61.",
    stats: [
      { label: "CV AUC", value: "0.88 ± 0.01", tone: "good" },
      { label: "Live AUC", value: "0.61", tone: "bad" },
    ],
    question: "What are the likely causes, in order?",
    answer:
      "The tight CV standard deviation is itself a clue — it means the estimate was *consistently* wrong, which points at something systematic rather than noise.\n\n**Leakage** is the first suspect. If a fitted step sat outside the CV loop, or a feature carries future information, every fold leaks identically and CV looks confident and excellent.\n\n**Train/serve skew** is second: the same feature computed differently offline and online. This is extremely common and produces exactly this pattern.\n\n**Distribution shift** is third: the population in production isn't the population in the training data, either because the training window was unrepresentative or because something changed.\n\n**An inappropriate CV splitter** is fourth: random K-fold on data with time or group structure gives an optimistic estimate for reasons that have nothing to do with the model.",
    nextSteps:
      "- Log the actual feature vectors used at inference and compare their distributions to training. This finds skew and shift fast.\n- Confirm every fitted transformation is inside the Pipeline passed to cross_val_score.\n- Re-validate with a time-based split against a genuinely later period.\n- Check for nulls or defaults appearing in production features that never appeared in training.",
  },
  {
    id: "s7",
    title: "The slow decay",
    setup:
      "A model launches at 0.84 AUC. Three months later it's at 0.71. Nothing in the code has changed.",
    question: "What's happening, and how should this have been caught earlier?",
    answer:
      "Some form of **drift**. Two flavours worth distinguishing:\n\n**Data drift** — the input distribution moved. New customer segments, a new acquisition channel, a changed upstream schema, seasonality the training window didn't cover.\n\n**Concept drift** — the relationship between features and target moved. This is the more serious one, and it's guaranteed in adversarial domains: fraud tactics evolve specifically to defeat the model you deployed.\n\nThere's also a third possibility worth ruling out: a **feedback loop**. If the model's decisions change what data you collect — blocking transactions you never learn the outcome of — the live distribution is partly the model's own doing.",
    nextSteps:
      "- Monitoring should have caught this. Track per-feature distribution shift (PSI or KS) and prediction distribution shift, both of which are available immediately, without waiting for labels.\n- Track live performance once ground truth arrives, accepting the label delay.\n- Define a retraining trigger — schedule plus drift threshold plus performance threshold — before deployment, not after.\n- For adversarial domains, plan for frequent retraining as a baseline, not as an incident response.",
  },
  {
    id: "s8",
    title: "Offline win, online loss",
    setup:
      "A new recommender improves offline NDCG@10 by 6%. In the A/B test, click-through rate drops by 2%.",
    stats: [
      { label: "Offline NDCG@10", value: "+6%", tone: "good" },
      { label: "Online CTR", value: "−2%", tone: "bad" },
    ],
    question: "How is that possible?",
    answer:
      "Offline evaluation of a recommender is measured against logged data that the **old** model generated. That creates several ways for offline and online to disagree.\n\nThe logged data only contains feedback on items the old system chose to show, so the offline metric rewards a model that reproduces the old model's choices — a genuinely different, better model can score worse offline. This is presentation bias, and it's the core difficulty in offline recommender evaluation.\n\nBeyond that: the offline metric may not be the business metric — NDCG rewards ranking quality on the items present in the log, and CTR depends on diversity, freshness and position effects too. And a model that concentrates on safe, popular items can look excellent offline while reducing discovery, which costs engagement.",
    nextSteps:
      "- Use counterfactual / off-policy evaluation (inverse propensity scoring) rather than naive NDCG on logged data.\n- Check for a serving difference: latency, candidate-generation changes, or a feature computed differently online.\n- Segment the A/B result — the loss may be concentrated in one user cohort.\n- Treat the online test as the ground truth. Offline metrics are a filter for what's worth testing, not a substitute for testing.",
  },
  {
    id: "s9",
    title: "The perfect feature",
    setup:
      "A churn model's single most important feature is `days_since_last_login`. For nearly every churned customer in the training data, its value is very large.",
    question: "Is this a good feature?",
    answer:
      "It depends entirely on **when it's measured**, and this is the question to ask out loud.\n\nIf it's computed as of the prediction timestamp — say, 30 days before the renewal date — then it's a legitimate and genuinely strong feature. Disengagement genuinely does precede churn, and that's useful signal.\n\nIf it's computed from a current snapshot of the database, it's **leakage**. Customers who churned months ago haven't logged in since, so the value is enormous purely *because* they churned. The model has learned a tautology, will score beautifully offline, and will be useless in production where every active customer has a small value.\n\nThere's also a subtler variant: even with correct timing, the feature can be **too close to the label** to be actionable. If it only becomes predictive two days before cancellation, the business has no time to intervene.",
    nextSteps:
      "- Establish the point-in-time definition of the feature and verify it against the prediction timestamp.\n- Check the distribution of the feature for non-churned customers in the same window — if it's tightly bimodal by label, be suspicious.\n- Ask what the intervention window is, and only use features available early enough for the intervention to be possible.\n- Validate on a later time period, where correct timing is enforced by construction.",
  },
  {
    id: "s10",
    title: "SMOTE helped the CV score",
    setup:
      "On a 2%-positive dataset, applying SMOTE lifted 5-fold CV F1 from 0.31 to 0.68. On the held-out test set, F1 is 0.29.",
    stats: [
      { label: "CV F1 before", value: "0.31", tone: "neutral" },
      { label: "CV F1 with SMOTE", value: "0.68", tone: "warn" },
      { label: "Test F1", value: "0.29", tone: "bad" },
    ],
    question: "What went wrong?",
    answer:
      "SMOTE was almost certainly applied **before** cross-validation rather than inside each training fold.\n\nWhen you oversample the whole training set first, synthetic points are interpolated between real minority points — including points that will later land in a validation fold. Each validation fold then contains synthetic near-copies of its own rows, generated from those rows. The model has effectively seen the validation data, and the CV F1 becomes a measurement of memorisation. The test set, untouched by SMOTE, reveals the truth.\n\nThe test score being roughly the same as the pre-SMOTE CV score is the tell: SMOTE didn't help at all, it just contaminated the estimate.",
    nextSteps:
      "- Put SMOTE inside an `imblearn.Pipeline` so it's applied only to each training fold, never to the validation fold.\n- Re-run CV and expect a much more modest number.\n- Try `class_weight='balanced'` plus threshold tuning first — on most tabular problems it matches or beats SMOTE with far less machinery.\n- Never resample the validation or test set. Evaluation must happen on the real distribution.",
  },
  {
    id: "s11",
    title: "The 10ms budget",
    setup:
      "Your best model is a random forest with 800 trees at depth 20. It hits the accuracy target. Serving requirement: **p99 latency under 10ms**, 5,000 requests per second.",
    question: "What do you do?",
    answer:
      "Accuracy isn't the binding constraint here — latency is. A model that misses the SLA is not deployable regardless of its AUC, so the honest answer starts by treating this as a systems problem rather than a modelling one.\n\nThe options, roughly in order of how much accuracy they cost:",
    nextSteps:
      "- **Shrink the ensemble.** Plot accuracy against tree count — the curve is usually flat well below 800. Depth 20 is also likely far more than needed.\n- **Switch to a faster family.** A well-tuned LightGBM with 200 shallow trees is often as accurate and dramatically faster than a deep forest.\n- **Distil.** Train a small model to mimic the big one's outputs; you often keep most of the accuracy.\n- **Fall back to a linear model** if the accuracy gap turns out to be small — sub-millisecond and trivially scalable.\n- **Change the serving pattern.** Precompute predictions in batch if inputs are known in advance; cache repeated queries.\n- **Quantify the tradeoff for the business.** \"We can hit 10ms with 0.8% less recall\" is a decision they should make, not you.",
  },
  {
    id: "s12",
    title: "Negative R²",
    setup: "A regression model reports R² = 0.92 on training data and **−0.15** on the test set.",
    stats: [
      { label: "Train R²", value: "0.92", tone: "good" },
      { label: "Test R²", value: "−0.15", tone: "bad" },
    ],
    question: "What does a negative R² mean, and what's the diagnosis?",
    answer:
      "Negative R² means the model performs **worse than simply predicting the mean of the target**. R² compares your residual sum of squares to that of the constant mean predictor; below zero, you'd have been better off with the constant.\n\nCombined with 0.92 on training, this is severe **overfitting** — the model has fitted training noise so aggressively that its predictions on new data are actively harmful.\n\nWorth ruling out two alternatives: a **distribution shift** between train and test (if the split was random on temporal data, or if the test set came from a different source), and a **bug** — a scaling or inverse-transform applied to one set and not the other will produce exactly this.",
    nextSteps:
      "- Plot predicted vs actual on the test set. Overfitting and a transform bug look completely different there.\n- Compare feature distributions between train and test.\n- Heavily regularize — Ridge with a cross-validated alpha, or a constrained GBM — and re-measure.\n- Check the row count. Negative test R² is common when the model has many parameters relative to the data.",
  },
  {
    id: "s13",
    title: "Two models, two metrics",
    setup:
      "Model A: ROC-AUC 0.89, precision@100 = 0.42. Model B: ROC-AUC 0.85, precision@100 = 0.61. Your analysts can review exactly 100 cases per day.",
    stats: [
      { label: "A · AUC", value: "0.89", tone: "good" },
      { label: "A · P@100", value: "0.42", tone: "warn" },
      { label: "B · AUC", value: "0.85", tone: "neutral" },
      { label: "B · P@100", value: "0.61", tone: "good" },
    ],
    question: "Which do you ship?",
    answer: 
      "**Model B**, and the reasoning is that precision@100 is the metric that matches the actual operating constraint.\n\nThe analysts review 100 cases a day, full stop. Model B puts 61 real cases in front of them; Model A puts 42. In terms of what the system actually delivers, B is roughly 45% more productive, and A's better AUC is describing quality across a region of the curve nobody operates in.\n\nThis is the general lesson: **AUC summarises the whole curve, but you deploy one point on it.** When the operating point is fixed by a hard capacity constraint, evaluate at that point.",
    nextSteps:
      "- Confirm the capacity is genuinely fixed. If it could flex, model the tradeoff — A might overtake B at a higher K.\n- Check the recall implication too: catching 61 versus 42 of how many total positives?\n- Consider whether A's better ranking could be salvaged by recalibration or a different threshold — sometimes the gap is an artefact of score distribution rather than ranking.\n- Report both numbers to stakeholders. The single-metric framing is what created the confusion.",
  },
  {
    id: "s14",
    title: "Random folds on a time series",
    setup:
      "A demand forecasting model is validated with random 5-fold cross-validation. CV MAPE is 4.2%. Deployed, it runs at 19%.",
    stats: [
      { label: "CV MAPE", value: "4.2%", tone: "good" },
      { label: "Live MAPE", value: "19%", tone: "bad" },
    ],
    question: "What's the problem?",
    answer:
      "The validation strategy is invalid for this data. Random K-fold on a time series means that for any given fold, the training data includes points from **after** the validation points. The model gets to see the future.\n\nThat helps enormously in two ways it can't reproduce in production: it learns the level and trend of the exact period it's being scored on, and any lag or rolling feature computed across the whole series carries information from the validation window. Neither is available when you're genuinely forecasting forward.\n\nThe CV MAPE of 4.2% wasn't a bad estimate — it was an estimate of a completely different, much easier problem.",
    nextSteps:
      "- Switch to `TimeSeriesSplit` with an expanding or rolling window: always train on the past, validate on the future.\n- Insert a **gap** between train and validation equal to the longest feature lookback plus the forecast horizon, so windows can't overlap.\n- Recompute all lag and rolling features point-in-time.\n- Re-baseline expectations: the honest CV number will be much closer to 19%, and that's the number to improve from.",
  },
  {
    id: "s15",
    title: "Great overall, terrible for one group",
    setup:
      "A loan approval model has 0.87 AUC overall. Sliced by applicant age band, AUC is 0.89 for applicants over 35 and **0.58** for applicants under 25, who are 11% of the volume.",
    stats: [
      { label: "Overall AUC", value: "0.87", tone: "good" },
      { label: "Under 25", value: "0.58", tone: "bad" },
      { label: "Share of volume", value: "11%", tone: "neutral" },
    ],
    question: "Is this shippable?",
    answer:
      "The aggregate metric is hiding a group the model essentially cannot score. At 0.58 AUC, decisions for under-25s are close to arbitrary — and in lending, that has legal and ethical weight as well as commercial.\n\nThe likely causes: **insufficient representation** in training (11% of volume may be far less in the positive class), and **feature availability** — young applicants have thin credit files, so the features the model relies on are mostly missing or uninformative for them. A model trained to optimise aggregate performance will happily sacrifice a small subgroup to gain on the majority; nothing in the loss function objects.",
    nextSteps:
      "- Always slice evaluation by the segments that matter — demographic, geographic, product, channel, tenure. Aggregate metrics hide this by design.\n- Check whether the subgroup has enough positive examples to learn from; consider reweighting or a segment-specific model.\n- Look for features that are systematically missing for the subgroup, and find alternatives that exist for thin-file applicants.\n- Involve compliance early. Disparate performance across a protected or proxy characteristic is a regulatory issue, not just a modelling one.\n- Consider whether the model should abstain for this group and route to manual review rather than issuing a near-random decision.",
  },
  {
    id: "s16",
    title: "The exhaustive search",
    setup:
      "On an 800-row dataset, someone runs a grid search over 12,000 hyperparameter configurations with 5-fold CV. Best CV accuracy: 0.94. Test accuracy: 0.79.",
    stats: [
      { label: "Configs tried", value: "12,000", tone: "warn" },
      { label: "Best CV", value: "0.94", tone: "warn" },
      { label: "Test", value: "0.79", tone: "bad" },
    ],
    question: "What happened?",
    answer:
      "**Overfitting to the validation folds** — the multiple-comparisons problem applied to hyperparameter search.\n\nWith 640 training rows and 160 per fold, a single CV estimate carries a few points of noise. Evaluate 12,000 configurations against that same noisy estimate and the winner is selected partly for genuine merit and partly for having got lucky on these specific folds. The maximum of 12,000 noisy draws is systematically higher than the true best — that's a selection effect, not a modelling result. The 15-point drop to test is that optimism being removed.\n\nThe irony is that the search was too thorough. A smaller search would have produced a less biased CV estimate and probably a similar or better test score.",
    nextSteps:
      "- Shrink the search space dramatically. On 800 rows, a handful of well-chosen configurations beats an exhaustive grid.\n- Use repeated CV to reduce the noise in each estimate before comparing.\n- Prefer the simplest configuration whose CV score is within one standard error of the best — the 'one-standard-error rule'.\n- If you need an unbiased estimate of the tuned pipeline, use nested CV.\n- Treat 0.79 as the honest number and don't re-tune against the test set to recover the difference.",
  },
];
