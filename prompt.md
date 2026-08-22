Build a polished, interactive website for **Machine Learning Fundamentals interview preparation for an AI Engineer role**.

The website must be designed so that I can complete a **comprehensive review in approximately 4 hours**. It should focus on the concepts most likely to appear in **AI Engineer / Machine Learning Engineer interviews**, not on academic proofs or overly theoretical material.

The target learner already knows basic programming and has studied machine learning before, but needs a **fast, structured interview refresher**.

## Primary Goal

Create a self-contained website that helps me review the most important Machine Learning fundamentals in about **4 hours**.

The website should help me:

* Refresh core ML concepts quickly.
* Understand the intuition behind each concept.
* Remember important formulas.
* Recognize common interview questions.
* Compare similar concepts.
* Identify common mistakes and misconceptions.
* Practice short interview-style questions.
* Finish with a final rapid-review cheat sheet.

The website should be practical, concise, visual, and interview-focused.

---

# 1. Website Structure

Create the following main sections in this exact order:

1. **4-Hour Study Plan**
2. **ML Problem Types**
3. **Train / Validation / Test Split**
4. **Data Leakage**
5. **Overfitting and Underfitting**
6. **Bias–Variance Tradeoff**
7. **Model Evaluation Metrics**
8. **Cross-Validation**
9. **Feature Engineering**
10. **Feature Scaling**
11. **Handling Missing Data**
12. **Categorical Variables**
13. **Class Imbalance**
14. **Regularization**
15. **Gradient Descent**
16. **Hyperparameters vs Parameters**
17. **Common ML Algorithms**
18. **Model Selection**
19. **Ensemble Methods**
20. **Dimensionality Reduction**
21. **Clustering Fundamentals**
22. **ML Pipeline**
23. **Common Interview Questions**
24. **Rapid-Fire Quiz**
25. **Final Cheat Sheet**

Include a persistent sidebar or navigation menu so I can jump directly to any section.

---

# 2. 4-Hour Study Plan

At the top of the website, create a visual study schedule dividing the content into approximately:

### Hour 1 — Core Concepts

* ML problem types
* train/validation/test
* data leakage
* overfitting
* underfitting
* bias vs variance
* cross-validation

### Hour 2 — Metrics and Data Preparation

* classification metrics
* regression metrics
* feature scaling
* missing values
* categorical features
* class imbalance
* feature engineering

### Hour 3 — Models and Training

* linear regression
* logistic regression
* decision trees
* random forests
* gradient boosting
* SVM
* KNN
* Naive Bayes
* regularization
* gradient descent

### Hour 4 — Interview Review

* model comparison
* ensembles
* PCA
* clustering
* ML pipeline
* common interview questions
* rapid-fire quiz
* final cheat sheet

Show estimated time for each section.

Add progress tracking so I can mark each section as completed.

---

# 3. Teaching Style

For every concept, use this structure whenever appropriate:

### What it is

One concise definition.

### Intuition

Explain it in simple language.

### Example

Give a practical example.

### Interview Answer

Show how I could answer if an interviewer asks about the concept.

### Common Mistake

Explain typical misunderstandings.

### Key Takeaway

One or two sentences to memorize.

Avoid long textbook-style paragraphs.

Use short explanations, diagrams, tables, examples, and callout boxes.

---

# 4. ML Problem Types

Explain:

* Supervised learning
* Unsupervised learning
* Semi-supervised learning
* Self-supervised learning

For supervised learning, distinguish:

* Classification
* Regression

Give examples such as:

* Spam detection
* Credit risk
* House price prediction
* Customer churn
* Image classification

Include a comparison table:

| Type | Input | Label | Output Example |
| ---- | ----- | ----- | -------------- |

Include interview questions such as:

* What is the difference between supervised and unsupervised learning?
* Is anomaly detection supervised or unsupervised?
* Is recommendation a classification problem?

Explain that real systems can combine several ML problem types.

---

# 5. Train / Validation / Test Split

Explain clearly:

* Training set
* Validation set
* Test set

Use a diagram showing:

Dataset
→ Training
→ Validation
→ Test

Explain typical ratios such as:

* 80 / 10 / 10
* 70 / 15 / 15
* 80 / 20 when cross-validation is used

Explain:

* why the test set must remain untouched
* why validation data is used for hyperparameter tuning
* why evaluating repeatedly on the test set creates leakage

Include interview question:

> Why can't we tune hyperparameters using the test set?

Also cover:

* random split
* stratified split
* time-series split

Explain when random splitting is inappropriate.

---

# 6. Data Leakage

Make this an important highlighted section.

Explain:

* what data leakage is
* why it produces unrealistically good validation/test performance

Include examples:

* scaling before train/test split
* using future data
* target-derived features
* preprocessing using the entire dataset
* duplicate users appearing in both train and test sets

Include:

BAD:

Full dataset
→ normalize
→ split

GOOD:

Split
→ fit preprocessing on training data
→ apply to validation/test

Explain why `fit_transform()` should normally be applied only on training data and `transform()` on validation/test data.

Add interview questions.

---

# 7. Overfitting and Underfitting

Explain:

### Underfitting

* poor training performance
* poor validation performance
* model too simple

### Overfitting

* strong training performance
* weak validation performance
* model memorizes training data

Use simple plots showing:

* training error
* validation error
* model complexity

Explain ways to reduce overfitting:

* more training data
* regularization
* simpler model
* dropout for neural networks
* early stopping
* data augmentation
* cross-validation

Explain ways to reduce underfitting:

* more complex model
* better features
* train longer
* reduce regularization

Include a comparison table.

---

# 8. Bias–Variance Tradeoff

Explain:

High bias:

* overly simple model
* underfitting

High variance:

* overly sensitive to training data
* overfitting

Include a visual relationship:

Model complexity ↑

Bias ↓

Variance ↑

Explain the goal of balancing bias and variance.

Give examples:

* linear regression on complex nonlinear data → high bias
* deep decision tree → potentially high variance

Include interview question:

> What happens to bias and variance as model complexity increases?

---

# 9. Model Evaluation Metrics

Make this one of the largest sections.

## Classification Metrics

Explain the confusion matrix:

|                 | Predicted Positive | Predicted Negative |
| --------------- | ------------------ | ------------------ |
| Actual Positive | TP                 | FN                 |
| Actual Negative | FP                 | TN                 |

Explain:

### Accuracy

Formula:

Accuracy = (TP + TN) / Total

Discuss when accuracy is misleading.

### Precision

Precision = TP / (TP + FP)

Explain:

"Of everything predicted positive, how many were actually positive?"

Use example:

spam detection.

### Recall

Recall = TP / (TP + FN)

Explain:

"Of all actual positives, how many did we detect?"

Use example:

cancer detection.

### F1 Score

F1 = 2 × Precision × Recall / (Precision + Recall)

Explain when F1 is useful.

### Specificity

Specificity = TN / (TN + FP)

### ROC Curve

Explain:

* True Positive Rate
* False Positive Rate
* thresholds

### ROC-AUC

Explain what AUC means.

### Precision-Recall Curve

Explain why PR-AUC can be better than ROC-AUC for heavily imbalanced datasets.

Create a table:

| Scenario                    | Preferred Metric |
| --------------------------- | ---------------- |
| Balanced classes            | Accuracy         |
| Missing positives is costly | Recall           |
| False positives are costly  | Precision        |
| Need balance                | F1               |
| Strong imbalance            | PR-AUC           |

Include concrete interview scenarios.

Example:

Fraud detection:
Should we optimize precision or recall?

Explain that the answer depends on business cost.

---

# 10. Regression Metrics

Explain:

### MAE

MAE = mean(|y - ŷ|)

### MSE

MSE = mean((y - ŷ)²)

### RMSE

RMSE = √MSE

### R²

R² = 1 - SSres / SStot

Explain the intuition of each.

Compare MAE vs MSE:

* MSE penalizes large errors more heavily.
* MAE is more robust to outliers.

Include a compact comparison table.

---

# 11. Cross-Validation

Explain K-Fold cross-validation visually.

Example with 5 folds:

Fold 1: Validation
Fold 2–5: Training

Then rotate.

Explain:

* K-Fold
* Stratified K-Fold
* Leave-One-Out
* Time Series Split

Explain why cross-validation helps estimate generalization performance.

Explain why test data should still remain separate.

Include interview question:

> If you're using cross-validation, do you still need a test set?

Answer clearly.

---

# 12. Feature Engineering

Explain:

* creating useful features
* transforming raw data
* domain knowledge

Examples:

Date:
`2026-08-22`

Can produce:

* day
* month
* weekday
* weekend
* quarter

Explain:

* interaction features
* polynomial features
* log transformation
* binning
* aggregated features

Explain dangers of creating leakage.

---

# 13. Feature Scaling

Explain:

### Standardization

z = (x - mean) / standard deviation

### Min-Max Normalization

x' = (x - min) / (max - min)

Explain which algorithms care about scaling:

Scaling important:

* KNN
* SVM
* Logistic Regression
* Linear Regression with regularization
* Neural Networks
* PCA

Usually not required:

* Decision Trees
* Random Forests
* Gradient Boosted Trees

Include a comparison table.

Explain why tree models do not generally need feature scaling.

---

# 14. Missing Data

Explain common approaches:

* remove rows
* remove columns
* mean imputation
* median imputation
* mode imputation
* model-based imputation
* add missing indicator

Discuss when missing values themselves contain information.

Example:

Income missing could indicate something about the user.

Explain why preprocessing must be learned only from training data.

---

# 15. Categorical Variables

Explain:

* label encoding
* ordinal encoding
* one-hot encoding
* target encoding
* embeddings

Explain when each is appropriate.

Highlight:

Do not arbitrarily label encode unordered categories such as:

Red = 1
Green = 2
Blue = 3

because it introduces a false numerical ordering.

---

# 16. Class Imbalance

Explain the problem.

Example:

99% legitimate transactions
1% fraud

Show why 99% accuracy can be useless.

Discuss:

* undersampling
* oversampling
* SMOTE
* class weights
* threshold tuning
* precision/recall
* F1
* PR-AUC

Explain why changing the classification threshold can trade precision for recall.

---

# 17. Regularization

Explain:

## L1 Regularization

Loss + λ Σ|w|

Characteristics:

* can push weights exactly to zero
* performs feature selection
* sparse models

## L2 Regularization

Loss + λ Σw²

Characteristics:

* shrinks weights
* generally does not make them exactly zero

Create a table comparing:

* L1
* L2
* Elastic Net

Explain:

Increasing λ generally:

* increases bias
* decreases variance

Connect regularization back to overfitting.

---

# 18. Gradient Descent

Explain:

1. Initialize parameters
2. Make predictions
3. Calculate loss
4. Calculate gradients
5. Update parameters
6. Repeat

Formula:

θ = θ - η∇J(θ)

Explain:

* θ = model parameters
* η = learning rate
* gradient = direction of increasing loss

Explain why we subtract the gradient.

Compare:

* Batch Gradient Descent
* Stochastic Gradient Descent
* Mini-batch Gradient Descent

Explain learning rate problems:

Too high:

* unstable
* overshoots minimum

Too low:

* slow convergence

Include simple loss-curve diagrams.

---

# 19. Parameters vs Hyperparameters

Explain clearly:

Parameters learned by the model:

* linear regression coefficients
* neural network weights
* decision tree split parameters

Hyperparameters chosen by us:

* learning rate
* tree depth
* number of estimators
* regularization strength
* batch size

Include table:

| Parameters | Hyperparameters |
| ---------- | --------------- |

Explain hyperparameter tuning techniques:

* grid search
* random search
* Bayesian optimization

Explain why random search can outperform grid search when only a few hyperparameters matter.

---

# 20. Common ML Algorithms

Create a separate card for each algorithm.

For every algorithm include:

* What it does
* Main intuition
* Strengths
* Weaknesses
* Important hyperparameters
* Scaling required?
* Handles nonlinear relationships?
* Interview questions

Cover:

## Linear Regression

Explain:

y = w₀ + w₁x₁ + ... + wₙxₙ

Discuss:

* coefficients
* assumptions
* MSE loss

## Logistic Regression

Explain:

sigmoid function

p = 1 / (1 + e^-z)

Explain why logistic regression is used for classification despite the name.

## Decision Tree

Explain:

recursive splitting.

Discuss:

* Gini impurity
* entropy
* information gain
* max depth
* overfitting

## Random Forest

Explain:

many decision trees trained using:

* bootstrap samples
* random feature subsets

Explain why random forests reduce variance.

## Gradient Boosting

Explain:

models trained sequentially.

Each new model attempts to correct errors of previous models.

Mention:

* XGBoost
* LightGBM
* CatBoost

Explain difference:

Random Forest = parallel independent trees.

Boosting = sequential trees correcting errors.

## K-Nearest Neighbors

Explain:

* choose K
* calculate distance
* vote / average

Discuss:

* feature scaling
* curse of dimensionality
* inference cost

## Support Vector Machine

Explain:

maximize margin between classes.

Discuss:

* support vectors
* kernel trick
* linear vs RBF kernel
* scaling requirement

## Naive Bayes

Explain:

Bayes theorem and conditional independence assumption.

Use:

* spam classification
* text classification

Mention why it can work well despite the unrealistic independence assumption.

---

# 21. Model Comparison Table

Create a large comparison table similar to:

| Algorithm | Classification | Regression | Scaling | Nonlinear | Handles Large Data | Interpretable |
| --------- | -------------- | ---------- | ------- | --------- | ------------------ | ------------- |

Cover:

* Linear Regression
* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* KNN
* SVM
* Naive Bayes
* Neural Networks

---

# 22. Model Selection

Explain how to choose a model.

Use cases such as:

### Need interpretability

* Linear regression
* Logistic regression
* shallow decision tree

### Tabular structured data

* Gradient boosting
* Random forest

### Small high-dimensional datasets

* SVM

### Image/text/complex unstructured data

* Neural networks

Explain that there is no universally best model.

Discuss the importance of:

* business constraints
* interpretability
* latency
* memory
* training time
* inference time
* accuracy

---

# 23. Ensemble Methods

Explain:

### Bagging

Models trained independently.

Example:

Random Forest.

Goal:

Reduce variance.

### Boosting

Models trained sequentially.

Example:

XGBoost.

Goal:

Reduce bias and improve predictions.

### Stacking

Predictions from multiple models are fed into another model.

Create a visual comparison.

---

# 24. Dimensionality Reduction

Explain PCA.

Cover intuition:

PCA creates new directions that capture maximum variance.

Explain:

* principal components
* variance explained
* dimensionality reduction
* decorrelation

Explain why scaling is usually necessary before PCA.

Explain advantages:

* visualization
* faster training
* noise reduction

Explain disadvantages:

* loss of interpretability
* information loss

Do not go deeply into eigenvalue derivations.

---

# 25. Clustering Fundamentals

Explain:

## K-Means

Steps:

1. Select K centroids
2. Assign points to nearest centroid
3. Recompute centroids
4. Repeat

Explain:

* choosing K
* elbow method
* sensitivity to scaling
* sensitivity to initialization
* problems with irregularly shaped clusters

Briefly mention:

* hierarchical clustering
* DBSCAN

Explain when clustering is useful.

---

# 26. ML Pipeline

Create a full end-to-end ML pipeline diagram:

Raw Data
↓
Data Cleaning
↓
Train/Test Split
↓
Feature Engineering
↓
Preprocessing
↓
Model Training
↓
Validation
↓
Hyperparameter Tuning
↓
Final Evaluation
↓
Deployment
↓
Monitoring

Explain each stage.

Add an interview question:

> Walk me through how you would build a machine learning model from raw data to production.

Provide a strong 60–90 second interview answer.

---

# 27. Important Interview Questions

Create at least **40 interview questions**.

Group them into:

### Fundamentals

Examples:

* What is the difference between classification and regression?
* What is overfitting?
* What is underfitting?
* Explain bias vs variance.
* What is data leakage?

### Metrics

* Precision vs recall?
* When would you use F1?
* ROC-AUC vs PR-AUC?
* Why can accuracy be misleading?

### Model Training

* What happens if learning rate is too high?
* Why do we use regularization?
* L1 vs L2?
* Why do we need validation data?

### Algorithms

* Random forest vs gradient boosting?
* Why don't trees require feature scaling?
* Logistic regression vs SVM?
* KNN advantages and disadvantages?
* Why is Naive Bayes called naive?

### Practical ML

* What would you do with an imbalanced dataset?
* How would you detect overfitting?
* What would you do if train accuracy is high but validation accuracy is low?
* What if both are low?
* How do you handle missing values?
* How do you handle categorical features?

Each question should have:

* a short answer
* a stronger interview answer
* an optional deeper explanation

Allow answers to be collapsed/expanded.

---

# 28. Scenario-Based Interview Questions

Include practical scenarios such as:

### Scenario 1

A fraud classifier gives:

99% accuracy
60% recall

Ask:

Is this a good model?

Explain the answer.

### Scenario 2

Training accuracy = 99%

Validation accuracy = 75%

Ask:

What is happening?

Answer:

likely overfitting.

Ask what to try next.

### Scenario 3

Training accuracy = 65%

Validation accuracy = 63%

Ask what might be happening.

Answer:

underfitting/high bias.

### Scenario 4

After adding more features, test performance suddenly increases dramatically.

Ask:

What should you investigate?

Answer:

possible data leakage.

### Scenario 5

Medical screening system:

Which matters more: precision or recall?

Explain tradeoffs.

Include at least **15 scenarios**.

---

# 29. Rapid-Fire Quiz

Create at least **50 short questions**.

Examples:

"What metric measures TP / (TP + FN)?"

Answer:

Recall.

"What kind of regularization can force weights to zero?"

Answer:

L1.

"Which usually needs scaling: Random Forest or KNN?"

Answer:

KNN.

"What problem does bagging primarily reduce?"

Answer:

Variance.

Make questions interactive:

* Show Question
* Reveal Answer
* Next Question

Track score if practical.

---

# 30. Final Cheat Sheet

Create a final one-page-style review section.

Organize it into compact blocks.

Include:

## Core Concepts

Overfitting:

Train good
Validation bad

Underfitting:

Train bad
Validation bad

Bias:

model too simple

Variance:

model too sensitive

## Classification Metrics

Accuracy:

(TP + TN) / Total

Precision:

TP / (TP + FP)

Recall:

TP / (TP + FN)

F1:

harmonic mean of precision and recall

## Regularization

L1:

* sparse
* feature selection

L2:

* weight shrinkage

## Scaling

Needs scaling:

* KNN
* SVM
* Logistic Regression
* PCA
* Neural Networks

Usually does not:

* Decision Trees
* Random Forest
* Gradient Boosting Trees

## Algorithms

Linear Regression:
regression

Logistic Regression:
classification

Decision Tree:
interpretable but can overfit

Random Forest:
bagging

Gradient Boosting:
sequential error correction

KNN:
distance based

SVM:
maximum margin

Naive Bayes:
probabilistic

## Key Relationships

More model complexity:

Bias ↓
Variance ↑

More regularization:

Bias ↑
Variance ↓

Higher classification threshold:

Precision generally ↑
Recall generally ↓

Lower classification threshold:

Recall generally ↑
Precision generally ↓

---

# 31. Visual Design

Use a modern technical-learning design.

Requirements:

* desktop-first but fully responsive
* clean typography
* comfortable reading width
* sticky sidebar navigation
* progress indicator
* section completion checkboxes
* collapsible interview answers
* searchable content if practical
* dark mode toggle
* code-style formatting for formulas where appropriate

Use subtle visual distinctions for:

* Definition
* Interview Tip
* Common Mistake
* Memorize This
* Example

Avoid excessive animation.

Prioritize speed and readability.

---

# 32. Interactive Visualizations

Create lightweight visual demonstrations where useful.

Examples:

### Bias–Variance

Slider:

Simple model ←→ Complex model

Show:

Bias decreases
Variance increases

### Classification Threshold

Slider:

0.1 → 0.9

Show conceptual effect:

threshold ↑

Precision ↑
Recall ↓

### Overfitting

Show curves for:

* training error
* validation error

as model complexity increases.

### Gradient Descent

Simple graph showing a point moving toward the minimum.

### K-Means

Simple animated demonstration of centroid assignment if practical.

Interactions should improve understanding rather than exist purely for decoration.

---

# 33. Interview Priority Labels

Label concepts with priority:

🔴 **Must Know**

🟠 **Important**

🟢 **Good to Know**

Use approximately:

70% Must Know
25% Important
5% Good to Know

The website is for a **4-hour review**, so do not let lower-priority material overwhelm the essential concepts.

---

# 34. Content Depth

For Must Know topics:

Provide enough detail for me to confidently explain the concept in an interview.

For Important topics:

Provide concise but meaningful understanding.

For Good to Know:

Keep the explanation brief.

Avoid mathematical proofs unless they genuinely help interview understanding.

The focus should be:

**intuition + practical application + interview explanation.**

---

# 35. Technical Requirements

Build the website as a modern frontend application.

Preferred stack:

* React
* TypeScript
* Tailwind CSS

Use reusable components such as:

* `TopicSection`
* `DefinitionCard`
* `InterviewQuestion`
* `QuizCard`
* `ComparisonTable`
* `FormulaCard`
* `ProgressTracker`

Store learning progress locally using `localStorage`.

No backend is required.

The site should run locally with standard commands such as:

```bash
npm install
npm run dev
```

Keep dependencies minimal.

---

# 36. Accuracy Requirements

All machine learning explanations must be technically correct.

Do not oversimplify concepts to the point of becoming inaccurate.

Where an interview answer depends on context, explicitly say:

"It depends on the cost of false positives versus false negatives."

Avoid misleading rules such as:

"Always use recall for fraud detection."

Instead explain the tradeoff.

---

# 37. Final User Experience

When I open the website, I should immediately see:

**Machine Learning Fundamentals — 4-Hour AI Engineer Interview Review**

Then:

* overall progress
* estimated time remaining
* 4-hour study plan
* Start Review button

At the end, show:

**Interview Readiness Check**

with approximately 15 statements such as:

* I can explain overfitting and underfitting.
* I can explain bias vs variance.
* I can choose between precision and recall.
* I understand train/validation/test splitting.
* I can explain L1 vs L2.
* I can compare random forest and gradient boosting.
* I can explain why feature scaling matters.
* I can describe a complete ML pipeline.

Allow each one to be checked.

Finish with:

**If you can explain every red-priority topic without looking at the notes, you are ready for the Machine Learning fundamentals portion of an AI Engineer interview.**
