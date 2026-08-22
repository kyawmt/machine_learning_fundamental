import { AlgorithmCard, type AlgorithmSpec } from "../components/AlgorithmCard";
import { Callout } from "../components/Callout";
import { DefinitionCard } from "../components/DefinitionCard";
import { Code, ComparisonTable, Formula, SubHeading, YesNo } from "../components/ui";
import { SigmoidCurve } from "../components/viz/diagrams";

const ALGORITHMS: AlgorithmSpec[] = [
  /* ---------------------------------------------------------------- */
  {
    id: "algo-linear-regression",
    name: "Linear Regression",
    priority: "must",
    task: "Regression",
    scaling: "not-required",
    nonlinear: "with-features",
    interpretable: "high",
    whatItDoes: (
      <>
        Fits a straight line (or hyperplane) through the data by choosing the coefficients that minimise the sum of
        squared residuals.
      </>
    ),
    formula: `ŷ = w₀ + w₁x₁ + w₂x₂ + … + wₙxₙ

Loss (MSE) = (1/n) Σ (yᵢ − ŷᵢ)²

Closed form (OLS):  w = (XᵀX)⁻¹ Xᵀy`,
    formulaCaption: (
      <>
        There's an exact solution — no iteration needed. Gradient descent is used anyway when{" "}
        <Code>XᵀX</Code> is too large to invert or is singular.
      </>
    ),
    intuition:
      "Each coefficient answers one question: holding everything else fixed, how much does the prediction move when this feature goes up by one unit? That direct readability is the reason linear regression survives in regulated industries.",
    strengths: [
      "Fast to train, trivial to serve, and it extrapolates in a predictable way.",
      "Coefficients are directly interpretable, with standard errors and p-values if you want inference.",
      "Convex loss with a closed-form solution — no local minima, no tuning to get it to converge.",
      "A strong baseline that tells you immediately whether the problem has linear structure.",
    ],
    weaknesses: [
      "Can only represent additive, linear effects — no interactions or curvature unless you engineer them in.",
      "Sensitive to outliers, because squared error lets one extreme point dominate.",
      "Multicollinearity makes coefficients unstable and un-interpretable, even when predictions stay fine.",
      "Assumes roughly constant error variance; heteroscedasticity breaks the inference (though not necessarily the prediction).",
    ],
    hyperparams: ["fit_intercept", "alpha (Ridge/Lasso)", "l1_ratio (ElasticNet)"],
    extra: (
      <Callout variant="tip" title="The assumptions, in the order they get asked">
        <ul className="bullet-list">
          <li><strong>Linearity</strong> — the relationship between features and target is linear in the parameters.</li>
          <li><strong>Independence</strong> — residuals are uncorrelated with each other (violated by time series).</li>
          <li><strong>Homoscedasticity</strong> — residual variance is constant across the range of predictions.</li>
          <li><strong>Normality of residuals</strong> — needed for p-values and confidence intervals, <em>not</em> for the point predictions.</li>
          <li><strong>No perfect multicollinearity</strong> — otherwise <Code>XᵀX</Code> isn't invertible.</li>
        </ul>
        <p className="mt-2">
          Note the fourth one carefully: people often say "linear regression assumes the data is normally
          distributed". It doesn't. It's an assumption about the <em>residuals</em>, and only for inference.
        </p>
      </Callout>
    ),
    questions: [
      {
        id: "lr-q1",
        q: "How do you interpret a coefficient of 2,500 on `square_feet`?",
        short: "Holding all other features constant, one extra square foot is associated with $2,500 more predicted price.",
        strong:
          "Two caveats I'd add. First, 'holding everything else constant' is doing a lot of work — if square_feet is correlated with bedroom count, that ceteris paribus condition may not correspond to anything you can actually do. Second, it's an association, not a causal effect; the model has no idea whether adding a square foot would raise the price. And if the features weren't standardized, coefficient magnitudes across features aren't comparable, because they're in different units.",
      },
      {
        id: "lr-q2",
        q: "When would linear regression beat gradient boosting?",
        short: "When the relationship really is close to linear, when data is scarce, when you need to extrapolate, or when interpretability is a hard requirement.",
        strong:
          "Three concrete situations. With a few hundred rows, a GBM has enough capacity to memorise them and a regularized linear model will often generalise better. When you need to predict outside the range of the training data, trees are flat beyond their last split while a linear model keeps its slope — for dose-response or price extrapolation that matters. And in regulated settings — credit, insurance, clinical — you may need to justify every prediction to an auditor, which a linear model does natively. I'd also say: always fit the linear baseline anyway, because it tells you how much of the signal is simple.",
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  {
    id: "algo-logistic-regression",
    name: "Logistic Regression",
    priority: "must",
    task: "Classification (binary and multiclass)",
    scaling: "required",
    nonlinear: "with-features",
    interpretable: "high",
    whatItDoes: (
      <>
        Fits a linear function of the features and squashes it through the sigmoid to get a probability between 0
        and 1. The decision boundary is linear in feature space.
      </>
    ),
    formula: `z = w·x + b

p(y=1|x) = σ(z) = 1 / (1 + e^(−z))

log( p / (1−p) ) = w·x + b        ← linear in the LOG-ODDS

Loss = −(1/n) Σ [ y·log(p) + (1−y)·log(1−p) ]   (log loss)`,
    formulaCaption: "Trained with log loss, which is convex — unlike MSE composed with a sigmoid.",
    intuition:
      "It's a linear regression on the log-odds. The sigmoid is just the function that converts an unbounded score into a probability. That's the honest answer to 'why is it called regression when it does classification' — it regresses log-odds, then thresholds.",
    strengths: [
      "Outputs well-behaved probabilities that are usually reasonably calibrated out of the box.",
      "Coefficients are interpretable as log-odds; exponentiate for an odds ratio.",
      "Convex loss — a unique global optimum, and it trains in seconds on millions of rows.",
      "A very hard baseline to beat on high-dimensional sparse data such as bag-of-words text.",
    ],
    weaknesses: [
      "The decision boundary is linear; XOR-like structure needs explicit interaction features.",
      "Needs feature scaling in practice, because scikit-learn applies L2 regularization by default.",
      "Struggles with perfectly separable data — coefficients run off to infinity without regularization.",
      "Sensitive to strongly correlated features, which destabilise the coefficients.",
    ],
    hyperparams: ["C (inverse of λ)", "penalty (l1/l2/elasticnet)", "solver", "class_weight", "max_iter"],
    extra: (
      <div className="space-y-3">
        <SigmoidCurve />
        <Callout variant="memorize" title="Say this if asked why it's called regression">
          Because it fits a <strong>linear regression to the log-odds</strong>. The model is{" "}
          <Code>log(p/(1−p)) = w·x + b</Code> — genuinely a linear regression, just on a transformed scale. The
          sigmoid maps that back to a probability, and classification only happens when you apply a threshold at the
          very end. The threshold isn't part of the model.
        </Callout>
      </div>
    ),
    questions: [
      {
        id: "lg-q1",
        q: "Why not train logistic regression with mean squared error?",
        short: "MSE composed with a sigmoid is non-convex, and it produces vanishing gradients exactly where the model is most wrong.",
        strong:
          "Two reasons. Mathematically, MSE applied to a sigmoid output gives a non-convex loss surface with local minima, whereas log loss is convex and guarantees a unique optimum. Practically, the gradient of MSE through a sigmoid is proportional to σ'(z), which goes to zero when the sigmoid saturates — so a confidently wrong prediction produces almost no gradient and learning stalls. Log loss cancels that derivative: its gradient reduces to (p − y)·x, which is large precisely when the prediction is badly wrong. Log loss is also the maximum-likelihood estimator for a Bernoulli target, so it's the principled choice as well as the practical one.",
      },
      {
        id: "lg-q2",
        q: "Logistic regression vs linear SVM — how do you choose?",
        short: "Very similar decision boundaries. Choose logistic regression if you need probabilities; linear SVM if you want the maximum-margin boundary and don't need calibrated outputs.",
        strong:
          "They differ mainly in the loss. Logistic regression uses log loss, which never stops caring — even correctly classified points far from the boundary contribute a little gradient. Linear SVM uses hinge loss, which is exactly zero once a point is beyond the margin, so only the support vectors influence the fit. That makes SVM a bit more robust to outliers far on the correct side, and gives it a sparse solution. But SVM's decision_function is an uncalibrated score, not a probability — you'd need Platt scaling. In practice on linearly separable, high-dimensional data they perform very similarly, and I'd default to logistic regression for the probabilities and the speed.",
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  {
    id: "algo-decision-tree",
    name: "Decision Tree",
    priority: "must",
    task: "Classification and Regression",
    scaling: "not-required",
    nonlinear: "yes",
    interpretable: "high",
    whatItDoes: (
      <>
        Recursively splits the data on one feature at a time, greedily choosing at each node the split that most
        reduces impurity, until a stopping rule fires. Predictions come from the leaf a sample lands in.
      </>
    ),
    formula: `Gini(t)     = 1 − Σ pᵢ²
Entropy(t)  = − Σ pᵢ log₂ pᵢ

Information Gain = Impurity(parent)
                   − Σ (nⱼ/n) · Impurity(childⱼ)

Regression trees minimise variance (MSE) instead.`,
    formulaCaption:
      "Gini and entropy pick almost identical splits in practice; Gini is marginally cheaper because there's no logarithm. Do not lose sleep over the choice.",
    intuition:
      "A flowchart of yes/no questions, learned from data. Because each split is chosen greedily to separate the classes as much as possible right now, the tree carves the feature space into axis-aligned rectangles — which is also why it can't represent a diagonal boundary without a staircase of splits.",
    strengths: [
      "Captures nonlinearity and feature interactions automatically, with no feature engineering.",
      "Needs no scaling and handles mixed numeric and categorical data.",
      "Genuinely interpretable — you can print the tree and read the rules.",
      "Robust to monotone transformations and to outliers in the features (a split is a rank comparison).",
    ],
    weaknesses: [
      "Overfits aggressively if unconstrained — a full-depth tree can reach 100% training accuracy on almost anything.",
      "High variance: resample the data and the tree structure changes substantially.",
      "Greedy and myopic — it optimises one split at a time, so it can miss combinations a global search would find.",
      "Axis-aligned splits approximate diagonal boundaries with a staircase, which needs many splits.",
      "Biased toward high-cardinality features when using impurity-based importance.",
    ],
    hyperparams: ["max_depth", "min_samples_split", "min_samples_leaf", "max_features", "ccp_alpha", "criterion"],
    questions: [
      {
        id: "dt-q1",
        q: "How does a decision tree decide where to split?",
        short: "It tries every feature and every candidate threshold, and picks the split with the largest impurity reduction — Gini or entropy for classification, variance for regression.",
        strong:
          "At each node it enumerates candidate splits — for a numeric feature, the midpoints between sorted unique values — and for each one computes the weighted impurity of the two resulting children. The information gain is the parent's impurity minus that weighted child impurity, and it takes the split with the highest gain. Then it recurses. Two things worth naming: it's greedy, so it never reconsiders an earlier split in light of a later one; and it's why the algorithm is scale-invariant, since the whole thing depends only on the ordering of values, not their magnitudes.",
      },
      {
        id: "dt-q2",
        q: "How do you stop a decision tree overfitting?",
        short: "Constrain its growth (max_depth, min_samples_leaf), prune it back (ccp_alpha), or stop using a single tree and use an ensemble.",
        strong:
          "Pre-pruning means limiting growth up front — max_depth, min_samples_leaf, min_impurity_decrease. It's fast but the thresholds are somewhat arbitrary, and a greedy stop can miss a good split that only pays off two levels down. Post-pruning grows the full tree and then removes branches that don't earn their complexity — cost-complexity pruning via ccp_alpha, tuned by cross-validation — which is more principled. But honestly, the answer I'd give in production is that a single tree is rarely the right model. If I need a tree's flexibility I use a random forest or a GBM, which handle the variance problem structurally. A single tree is for when interpretability is the actual requirement.",
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  {
    id: "algo-random-forest",
    name: "Random Forest",
    priority: "must",
    task: "Classification and Regression",
    scaling: "not-required",
    nonlinear: "yes",
    interpretable: "medium",
    whatItDoes: (
      <>
        Trains many deep decision trees, each on a bootstrap sample of the rows, and each split considering only a
        random subset of the features. Predictions are averaged (regression) or voted (classification).
      </>
    ),
    formula: `Two sources of randomness:
  1. Bootstrap sample of rows   (~63.2% unique rows per tree)
  2. Random feature subset at every split
        √p    classification (scikit-learn's default)
        p/3   regression (the classic recommendation;
              scikit-learn now defaults to all p)

Var(average of B correlated predictors)
      =  ρσ²  +  (1 − ρ)σ² / B
         ↑                ↑
   floor set by      shrinks as you
   correlation ρ     add more trees`,
    formulaCaption:
      "That formula is the whole design. Adding trees drives the second term down; the random feature subsetting is there to drive ρ down, which lowers the floor.",
    intuition:
      "One deep tree is low-bias and high-variance — it's right on average but wildly unstable. Average hundreds of them and the instability cancels while the low bias survives. The random feature subsetting is the clever part: without it, one dominant feature would be the top split in every tree, the trees would be near-identical, and averaging highly correlated things barely reduces variance at all.",
    strengths: [
      "Excellent out-of-the-box accuracy on tabular data with essentially no tuning.",
      "Very hard to overfit by adding trees — more trees never hurts accuracy, only runtime.",
      "Free validation via the out-of-bag estimate: each tree can be scored on the ~37% of rows it didn't see.",
      "Trees are independent, so training parallelises perfectly across cores.",
      "Handles mixed types, missing-ish data and outliers with little fuss.",
    ],
    weaknesses: [
      "Large memory footprint and slower inference than a single model — hundreds of trees to traverse.",
      "Loses the single tree's interpretability; you get importances rather than rules.",
      "Usually a little behind a well-tuned gradient boosting model on tabular benchmarks.",
      "Extrapolates poorly — predictions are bounded by the range of training targets.",
      "Default impurity-based feature importance is biased toward high-cardinality and continuous features. Use permutation importance instead.",
    ],
    hyperparams: ["n_estimators", "max_features", "max_depth", "min_samples_leaf", "bootstrap", "class_weight"],
    questions: [
      {
        id: "rf-q1",
        q: "Why does a random forest use random feature subsets and not just bootstrapping?",
        short: "To decorrelate the trees. Averaging highly correlated models barely reduces variance; the feature subsetting is what makes the averaging pay off.",
        strong:
          "The variance of an average of B identically distributed predictors with pairwise correlation ρ is ρσ² + (1−ρ)σ²/B. The second term vanishes as you add trees, but the first doesn't — it's a floor set entirely by how correlated the trees are. With bootstrapping alone, if one feature is strongly predictive it becomes the root split in nearly every tree, so the trees look alike, ρ stays high, and you hit that floor quickly. Restricting each split to a random subset of features forces different trees to use different structures, which pushes ρ down and lowers the floor. That's why max_features is the hyperparameter that actually matters in a random forest.",
      },
      {
        id: "rf-q2",
        q: "What is the out-of-bag score?",
        short: "A free validation estimate. Each bootstrap sample leaves out about 37% of rows, so each row can be scored by the trees that never saw it.",
        strong:
          "Sampling N rows with replacement means a given row is left out of a given tree with probability (1 − 1/N)^N, which converges to 1/e ≈ 0.368. So roughly 37% of rows are out-of-bag for each tree. To score a row, you average only the trees for which it was out-of-bag — that's an honest held-out prediction. Aggregated across all rows, the OOB score is a decent substitute for cross-validation at zero extra cost, which is handy on small datasets. I'd still keep a separate test set, and I wouldn't rely on OOB when the data has grouping or time structure the bootstrap ignores.",
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  {
    id: "algo-gradient-boosting",
    name: "Gradient Boosting",
    priority: "must",
    task: "Classification and Regression",
    scaling: "not-required",
    nonlinear: "yes",
    interpretable: "medium",
    whatItDoes: (
      <>
        Builds trees <strong>sequentially</strong>. Each new tree is fitted to the negative gradient of the loss with
        respect to the current ensemble's predictions — for squared error, that's literally the residuals — and is
        added, scaled by a small learning rate.
      </>
    ),
    formula: `F₀(x) = a constant (e.g. the mean of y)

for m = 1 … M:
    rᵢ  = −∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)     ← pseudo-residuals
    hₘ  = fit a small tree to rᵢ
    Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)        ← η = learning_rate

Final prediction = F_M(x)`,
    formulaCaption:
      "It is gradient descent — but in function space. Each tree is one step, and the learning rate is the step size.",
    intuition:
      "A committee that hires specialists for its own mistakes. The first tree makes a rough prediction; the second tree is trained only on what the first got wrong; the third on what the pair still gets wrong. Each is deliberately weak — depth 3 to 6 — so the ensemble improves in small, controlled increments rather than lurching.",
    strengths: [
      "State of the art on tabular data. If a Kaggle-style tabular problem has a winner, it is usually a GBM.",
      "Handles nonlinearity and interactions natively; no scaling needed.",
      "Extremely flexible — swap the loss for regression, classification, ranking, quantiles, survival.",
      "Modern implementations handle missing values natively and train on millions of rows fast.",
    ],
    weaknesses: [
      "Will overfit if you keep adding trees. Early stopping on a validation set is not optional.",
      "More hyperparameters that genuinely interact — learning rate and n_estimators must be tuned together.",
      "Sequential, so training parallelises far less cleanly than a random forest.",
      "More sensitive to noisy labels than bagging, because it keeps chasing the examples it gets wrong.",
    ],
    hyperparams: [
      "learning_rate",
      "n_estimators",
      "max_depth / num_leaves",
      "subsample",
      "colsample_bytree",
      "min_child_weight",
      "reg_lambda",
      "early_stopping_rounds",
    ],
    extra: (
      <DefinitionCard term="XGBoost vs LightGBM vs CatBoost — one line each">
        <ul className="bullet-list">
          <li>
            <strong>XGBoost</strong> — uses a second-order (Newton) approximation of the loss and an explicitly
            regularized objective. Sparsity-aware, battle-tested, the default reference implementation.
          </li>
          <li>
            <strong>LightGBM</strong> — histogram-based binning and <em>leaf-wise</em> growth (it splits the leaf with
            the biggest gain, rather than growing level by level). Much faster on large data; more prone to
            overfitting on small data, so cap <Code>num_leaves</Code>.
          </li>
          <li>
            <strong>CatBoost</strong> — ordered target statistics for categorical features and ordered boosting to
            avoid the target leakage those statistics normally cause. The path of least resistance when you have many
            high-cardinality categoricals.
          </li>
        </ul>
      </DefinitionCard>
    ),
    questions: [
      {
        id: "gb-q1",
        q: "Random Forest vs Gradient Boosting — explain the difference.",
        short: "Random forest builds deep trees independently in parallel and averages them to cut variance. Boosting builds shallow trees sequentially, each correcting the ensemble's current errors, to cut bias.",
        strong:
          "The structural difference is independence versus dependence. In a random forest every tree is trained on its own bootstrap sample with no knowledge of the others, so training is embarrassingly parallel, and the trees are deliberately deep — low bias, high variance — because averaging is what kills the variance. In boosting each tree is trained on the residual errors of everything built so far, so it's inherently sequential, and the trees are deliberately shallow — high bias individually — because the ensemble's bias falls with every round. The practical consequences: random forests are much harder to overfit and need almost no tuning, so they're a great first model; GBMs usually win on accuracy but need early stopping and careful tuning of learning rate against number of trees. If someone told me I had one shot with no tuning budget, I'd take the random forest. With a tuning budget, LightGBM.",
        deeper:
          "One more distinction worth having: adding trees to a random forest is monotonically safe — accuracy converges and never degrades. Adding trees to a GBM eventually **increases** validation error, because the ensemble starts fitting noise in the residuals. That asymmetry is why `n_estimators` is a real hyperparameter for boosting and essentially just a compute budget for a forest.",
      },
      {
        id: "gb-q2",
        q: "What does the learning rate do in gradient boosting, and how does it relate to n_estimators?",
        short: "It scales each tree's contribution. Lower learning rate needs more trees, and the pair must be tuned together.",
        strong:
          "Each tree is added as η times its prediction, so η controls how much of the residual each round is allowed to absorb. A small η — 0.01 to 0.05 — means each tree makes a cautious correction, which generalises better because no single tree can overreact to noise, but you need many more rounds to reach the same fit. A large η reaches low training error quickly and overfits sooner. They trade off almost inversely: halving the learning rate roughly doubles the trees you need. My standard approach is to fix a small learning rate, set n_estimators deliberately too high, and let early stopping on a validation set choose the actual number. That way I'm only tuning one of the two.",
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  {
    id: "algo-knn",
    name: "K-Nearest Neighbors",
    priority: "must",
    task: "Classification and Regression",
    scaling: "required",
    nonlinear: "yes",
    interpretable: "medium",
    whatItDoes: (
      <>
        Stores the training set. To predict, find the K closest training points under some distance metric and take a
        majority vote (classification) or an average (regression).
      </>
    ),
    formula: `1. Choose K and a distance metric
2. Compute distance from the query to every training point
3. Take the K smallest
4. Classification → majority vote (optionally distance-weighted)
   Regression     → mean of their targets

Euclidean:  d = √Σ(xᵢ − x'ᵢ)²
Manhattan:  d = Σ|xᵢ − x'ᵢ|
Cosine:     d = 1 − (x·x')/(‖x‖‖x'‖)`,
    intuition:
      "There is no model. The training data is the model — this is the canonical 'lazy learner'. All the work happens at prediction time, which is the opposite of every other algorithm here and the reason its cost profile is unusual.",
    strengths: [
      "No training phase at all; adding data is just appending to the store.",
      "Naturally handles complex, non-parametric decision boundaries.",
      "Naturally multiclass, and the prediction is easy to explain: 'these five similar cases were mostly X'.",
      "The same idea underpins vector search and RAG retrieval — nearest neighbours over embeddings.",
    ],
    weaknesses: [
      "Inference is O(N·d) per query with a naive scan — expensive at scale and it grows with your dataset.",
      "Must store the entire training set in memory.",
      "Feature scaling is mandatory; without it the largest-unit feature decides every neighbour.",
      "Suffers badly from the curse of dimensionality — distances concentrate and 'nearest' stops meaning much.",
      "Sensitive to irrelevant features, which add noise to every distance computation.",
      "Struggles with class imbalance, since the majority class dominates most neighbourhoods.",
    ],
    hyperparams: ["n_neighbors (K)", "weights (uniform/distance)", "metric", "p (Minkowski power)", "algorithm (kd_tree/ball_tree/brute)"],
    extra: (
      <Callout variant="tip" title="Why this matters for an AI Engineer role">
        KNN is the conceptual core of <strong>vector search</strong>. A RAG system embeds documents, then retrieves
        the k nearest neighbours of the query embedding — usually with cosine distance, and using an{" "}
        <strong>approximate</strong> nearest-neighbour index (HNSW, IVF-PQ) because an exact scan over millions of
        vectors is too slow. If you're asked "how does retrieval work in your RAG pipeline?", the honest answer is
        "approximate KNN over embeddings", and every KNN weakness above — scaling, dimensionality, irrelevant
        dimensions, memory — shows up there too.
      </Callout>
    ),
    questions: [
      {
        id: "knn-q1",
        q: "What happens as you increase K?",
        short: "Bias goes up, variance goes down. K=1 fits every point exactly; very large K approaches predicting the global majority class.",
        strong:
          "K=1 gives a decision boundary that follows every training point, including the mislabelled ones — zero training error and high variance. As K grows you're averaging over a bigger neighbourhood, so the boundary smooths, individual noisy points stop mattering, and variance falls while bias rises. At K = N the prediction is the global majority regardless of input. I'd pick K by cross-validation, and for binary classification I'd use an odd K to avoid ties. Distance weighting is a nice middle ground — it lets you use a larger K while still letting closer neighbours count for more.",
      },
      {
        id: "knn-q2",
        q: "What is the curse of dimensionality and how does it affect KNN?",
        short: "In high dimensions all pairwise distances become similar, so 'nearest' stops being meaningful and the data becomes hopelessly sparse.",
        strong:
          "Two effects. First, volume grows exponentially with dimension, so any fixed number of points becomes vanishingly sparse — to keep the same density going from 10 to 20 dimensions you'd need the square of the data. Second, and worse for KNN, the ratio of the distance to the nearest point over the distance to the farthest point converges to 1 as dimension grows. Everything is roughly equidistant, so the neighbourhood you retrieve isn't meaningfully more similar than a random sample. Mitigations: reduce dimensions with PCA or a learned embedding, use cosine distance where magnitude is uninformative, do aggressive feature selection, or use a model that isn't distance-based. It's also why embedding quality matters so much in vector search — good embeddings put the useful signal into a low-dimensional manifold where distance means something again.",
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  {
    id: "algo-svm",
    name: "Support Vector Machine",
    priority: "important",
    task: "Classification (SVC) and Regression (SVR)",
    scaling: "required",
    nonlinear: "with-kernel",
    interpretable: "low",
    whatItDoes: (
      <>
        Finds the hyperplane that separates the classes with the <strong>largest possible margin</strong> — the
        widest gap between the boundary and the nearest points of each class.
      </>
    ),
    formula: `Maximise the margin  2/‖w‖
subject to   yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ,   ξᵢ ≥ 0

Equivalently, minimise:
    ½‖w‖²  +  C · Σ ξᵢ
      ↑            ↑
  wide margin   penalty for violations

RBF kernel:  K(x, x') = exp(−γ‖x − x'‖²)`,
    formulaCaption: (
      <>
        <Code>C</Code> is an inverse regularization strength: large C means violations are expensive, so the margin
        narrows and the model fits harder. <Code>γ</Code> controls how far a single point's influence reaches.
      </>
    ),
    intuition:
      "Don't just find a line that separates the classes — find the one with the most clearance on both sides, on the theory that a boundary sitting in the widest empty corridor is the one most likely to survive new data. Only the points on the edge of that corridor, the support vectors, determine the boundary. Move a point deep inside its own class and nothing changes.",
    strengths: [
      "Effective in high-dimensional spaces, including when features outnumber samples.",
      "The kernel trick gives nonlinear boundaries without ever materialising the high-dimensional features.",
      "Memory-efficient at prediction time — only the support vectors are stored.",
      "Maximum-margin objective has strong theoretical backing and works well on small, clean datasets.",
    ],
    weaknesses: [
      "Scales badly: roughly O(n²)–O(n³) in the number of samples. Impractical much beyond ~100k rows.",
      "No probability output natively — probabilities come from Platt scaling, which requires extra cross-validation.",
      "Requires scaling, and is quite sensitive to C and gamma, so tuning is mandatory.",
      "Hard to interpret once you use a nonlinear kernel.",
      "Struggles when classes overlap heavily and noise is high.",
    ],
    hyperparams: ["C", "kernel (linear/rbf/poly)", "gamma", "degree", "class_weight", "epsilon (SVR)"],
    extra: (
      <DefinitionCard term="The kernel trick, in one paragraph">
        <p className="mb-2">
          The SVM's optimisation only ever needs <strong>inner products</strong> between pairs of points, never the
          points themselves. A kernel function <Code>K(x, x')</Code> computes the inner product that{" "}
          <em>would</em> result from mapping both points into a much higher-dimensional space — without ever
          performing that mapping.
        </p>
        <Formula>{`Linear:  K(x,x') = x·x'
Poly:    K(x,x') = (x·x' + c)^d
RBF:     K(x,x') = exp(−γ‖x − x'‖²)      ← infinite-dimensional feature space`}</Formula>
        <p className="mt-2">
          <strong>gamma</strong> is the knob people ask about: high γ means each point's influence decays fast, so
          the boundary becomes tight and wiggly around individual points — overfitting. Low γ means broad influence
          and a nearly linear boundary — underfitting. C and gamma must be tuned jointly, usually on a log grid.
        </p>
      </DefinitionCard>
    ),
    questions: [
      {
        id: "svm-q1",
        q: "What is a support vector?",
        short: "A training point that lies on the margin boundary or violates it. Those are the only points that determine the decision boundary.",
        strong:
          "In the solved dual problem most training points get a dual coefficient of exactly zero and drop out entirely — the fitted boundary depends only on the handful with nonzero coefficients, and those are the support vectors. Practically that means you could delete every non-support-vector from the training set, refit, and get the identical model. It's also why SVMs are memory-efficient at inference: you only carry the support vectors. And it explains the robustness property — moving a point that sits comfortably inside its own class has no effect at all, whereas moving a support vector moves the boundary.",
      },
      {
        id: "svm-q2",
        q: "What does C control, and what happens at the extremes?",
        short: "C penalises margin violations. Large C → narrow margin, fits hard, risks overfitting. Small C → wide margin, tolerates errors, risks underfitting.",
        strong:
          "C is the weight on the slack term in the objective, so it's an inverse regularization strength. With C very large, misclassifying a training point is enormously expensive, so the optimiser will contort the boundary to get every point right — that's low bias and high variance, and on noisy data it chases outliers. With C very small, the optimiser prefers a wide margin and shrugs off violations, giving a smoother, more heavily regularized boundary — higher bias, lower variance. I tune it on a log scale, typically 0.01 to 100, jointly with gamma for an RBF kernel, since the two interact strongly.",
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  {
    id: "algo-naive-bayes",
    name: "Naive Bayes",
    priority: "important",
    task: "Classification",
    scaling: "not-required",
    nonlinear: "no",
    interpretable: "medium",
    whatItDoes: (
      <>
        Applies Bayes' theorem with a strong simplifying assumption: that every feature is conditionally independent
        of every other, given the class. That turns an intractable joint probability into a simple product.
      </>
    ),
    formula: `Bayes:      P(y|x) = P(x|y) · P(y) / P(x)

Naive assumption:
            P(x₁,…,xₙ | y) = Π P(xᵢ | y)

So:         ŷ = argmax_y  P(y) · Π P(xᵢ | y)

In practice, sum logs to avoid underflow:
            ŷ = argmax_y  [ log P(y) + Σ log P(xᵢ|y) ]`,
    formulaCaption:
      "P(x) is the same for every class, so it drops out of the argmax — which is why you don't need to compute it.",
    intuition:
      "Modelling the full joint distribution of features given a class is hopeless with any realistic amount of data. The naive assumption lets you estimate each feature's distribution separately — n small one-dimensional problems instead of one impossible n-dimensional one. The assumption is essentially always false, and the classifier works anyway.",
    strengths: [
      "Extremely fast to train and predict — one pass over the data, and it supports online updates.",
      "Works well with very little training data, because each parameter is estimated from a simple marginal count.",
      "Handles high-dimensional sparse data gracefully, which is why it's a classic text baseline.",
      "Naturally multiclass, with no extra machinery.",
      "Almost no hyperparameters to tune.",
    ],
    weaknesses: [
      "The independence assumption is essentially always violated, so the predicted probabilities are poorly calibrated — typically pushed toward 0 and 1.",
      "Correlated features get their evidence double-counted, which makes it overconfident.",
      "Zero-frequency problem: an unseen feature-class combination gives probability zero and wipes out the product. Laplace smoothing is required, not optional.",
      "Cannot learn feature interactions at all.",
      "Gaussian NB assumes each numeric feature is normally distributed within each class.",
    ],
    hyperparams: ["alpha (Laplace/Lidstone smoothing)", "fit_prior", "class_prior", "var_smoothing (Gaussian)"],
    extra: (
      <DefinitionCard term="Which variant to use">
        <ul className="bullet-list">
          <li>
            <strong>Multinomial NB</strong> — count data. The standard for bag-of-words and TF-IDF text
            classification.
          </li>
          <li>
            <strong>Bernoulli NB</strong> — binary presence/absence features. Explicitly models the absence of a
            term, which helps on short documents.
          </li>
          <li>
            <strong>Gaussian NB</strong> — continuous features, assuming a normal distribution per feature per class.
          </li>
          <li>
            <strong>Complement NB</strong> — a variant designed for imbalanced text data.
          </li>
        </ul>
      </DefinitionCard>
    ),
    questions: [
      {
        id: "nb-q1",
        q: "Why is Naive Bayes called 'naive', and why does it work anyway?",
        short: "Naive because it assumes features are conditionally independent given the class — nearly always false. It works because classification only needs the argmax to be right, not the probabilities.",
        strong:
          "In spam detection, 'free' and 'money' clearly co-occur, so treating them as independent double-counts their evidence. That makes the estimated probabilities badly miscalibrated — Naive Bayes is notoriously overconfident, spitting out 0.9999. But classification only depends on which class scores highest, and the double-counting tends to inflate the scores of competing classes in a broadly similar way, so the ranking often survives the distortion. There's a well-known result formalising this: Naive Bayes can be optimal for classification even when its probability estimates are far off. The practical takeaway is: use it as a fast, strong baseline for text, and don't use its probabilities for anything that depends on them being accurate.",
      },
      {
        id: "nb-q2",
        q: "What is Laplace smoothing and why do you need it?",
        short: "Adding a pseudo-count to every feature-class combination so an unseen combination doesn't produce a probability of zero and annihilate the whole product.",
        strong:
          "The prediction is a product of per-feature conditional probabilities. If a word never appeared in the training data for class 'spam', its estimated P(word | spam) is zero, and one zero drives the entire product to zero — that single unseen word vetoes the class no matter how much other evidence points to it. Laplace smoothing adds α (usually 1) to every count and adjusts the denominator accordingly, so every combination gets a small nonzero probability. In scikit-learn that's the `alpha` parameter, and setting alpha=0 is a reliable way to break your text classifier the first time it meets a new word.",
      },
    ],
  },
];

export function Algorithms() {
  return (
    <>
      <Callout variant="tip" title="How to study this section">
        Don't memorise the cards. For each algorithm, be able to answer four things in about twenty seconds:{" "}
        <strong>what it optimises</strong>, <strong>whether it needs scaling and why</strong>,{" "}
        <strong>whether it handles nonlinearity</strong>, and <strong>the one situation where you'd pick it</strong>.
        That's the shape of nearly every algorithm question you'll get.
      </Callout>

      <div className="space-y-4">
        {ALGORITHMS.map((a) => (
          <AlgorithmCard key={a.id} a={a} />
        ))}
      </div>

      <SubHeading id="model-comparison-table" priority="must">
        The comparison table
      </SubHeading>
      <ComparisonTable
        dense
        headers={[
          "Algorithm",
          "Classification",
          "Regression",
          "Needs scaling",
          "Nonlinear",
          "Large data",
          "Interpretable",
          "Training cost",
          "Inference cost",
        ]}
        rows={[
          ["Linear Regression", <YesNo key="1" v="no" />, <YesNo key="2" v="yes" />, <YesNo key="3" v="no" />, <YesNo key="4" v="no" />, <YesNo key="5" v="yes" />, "High", "Very low", "Very low"],
          ["Logistic Regression", <YesNo key="6" v="yes" />, <YesNo key="7" v="no" />, <YesNo key="8" v="yes" />, <YesNo key="9" v="no" />, <YesNo key="10" v="yes" />, "High", "Very low", "Very low"],
          ["Decision Tree", <YesNo key="11" v="yes" />, <YesNo key="12" v="yes" />, <YesNo key="13" v="no" />, <YesNo key="14" v="yes" />, <YesNo key="15" v="yes" />, "High", "Low", "Very low"],
          ["Random Forest", <YesNo key="16" v="yes" />, <YesNo key="17" v="yes" />, <YesNo key="18" v="no" />, <YesNo key="19" v="yes" />, <YesNo key="20" v="yes" />, "Medium", "Medium (parallel)", "Medium"],
          ["Gradient Boosting", <YesNo key="21" v="yes" />, <YesNo key="22" v="yes" />, <YesNo key="23" v="no" />, <YesNo key="24" v="yes" />, <YesNo key="25" v="yes" />, "Medium", "Medium (sequential)", "Medium"],
          ["KNN", <YesNo key="26" v="yes" />, <YesNo key="27" v="yes" />, <YesNo key="28" v="yes" />, <YesNo key="29" v="yes" />, <YesNo key="30" v="no" />, "Medium", "None", "High"],
          ["SVM", <YesNo key="31" v="yes" />, <YesNo key="32" v="yes" />, <YesNo key="33" v="yes" />, "With kernel", <YesNo key="34" v="no" />, "Low", "High", "Low–Medium"],
          ["Naive Bayes", <YesNo key="35" v="yes" />, <YesNo key="36" v="no" />, <YesNo key="37" v="no" />, <YesNo key="38" v="no" />, <YesNo key="39" v="yes" />, "Medium", "Very low", "Very low"],
          ["Neural Network", <YesNo key="40" v="yes" />, <YesNo key="41" v="yes" />, <YesNo key="42" v="yes" />, <YesNo key="43" v="yes" />, <YesNo key="44" v="yes" />, "Low", "High", "Medium–High"],
        ]}
        caption="“Large data” means it stays practical as rows grow into the millions. KNN and SVM are the two that genuinely fall over — KNN at inference time, SVM at training time."
      />

      <Callout variant="memorize" title="The one-line summary of each">
        <div className="grid gap-1.5 sm:grid-cols-2">
          <Line k="Linear Regression" v="Fits a line. Interpretable baseline for continuous targets." />
          <Line k="Logistic Regression" v="Linear in log-odds. Fast, calibrated, hard to beat on sparse text." />
          <Line k="Decision Tree" v="Readable rules. Overfits alone; the building block for everything below." />
          <Line k="Random Forest" v="Bagging. Averages deep trees to cut variance. Great default." />
          <Line k="Gradient Boosting" v="Sequential error correction. Usually the accuracy winner on tabular data." />
          <Line k="KNN" v="Distance-based, no training, expensive inference. The idea behind vector search." />
          <Line k="SVM" v="Maximum margin. Strong on small, high-dimensional, clean data." />
          <Line k="Naive Bayes" v="Probabilistic, assumes independence. Fast text baseline." />
          <Line k="Neural Network" v="Universal approximator. The answer for unstructured data — images, audio, text." />
        </div>
      </Callout>
    </>
  );
}

function Line({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex gap-2 text-[0.86rem]">
      <span className="min-w-[132px] shrink-0 font-bold text-ink">{k}</span>
      <span className="text-muted">{v}</span>
    </div>
  );
}
