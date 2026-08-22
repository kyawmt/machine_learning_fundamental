import type { QA } from "../components/InterviewQuestion";

export interface QAGroup {
  id: string;
  title: string;
  blurb: string;
  questions: QA[];
}

export const QA_GROUPS: QAGroup[] = [
  /* ================================================================ */
  {
    id: "qa-fundamentals",
    title: "Fundamentals",
    blurb: "The questions that open almost every ML screen. Answer these crisply and the rest of the interview goes better.",
    questions: [
      {
        id: "f1",
        q: "What is the difference between classification and regression?",
        short: "Classification predicts a discrete class label; regression predicts a continuous numeric value.",
        strong:
          "The difference propagates through the whole stack. Classification uses cross-entropy loss and is measured with precision, recall, F1 and AUC; regression uses squared or absolute error and is measured with MAE, RMSE and R². The decision that matters is which one the downstream system needs — if someone acts on a yes/no, classify and give them a calibrated probability so they can set the threshold; if the magnitude feeds a calculation, regress. Ordinal targets like a 1–5 rating sit in between and can reasonably be modelled either way.",
        ref: "problem-types",
      },
      {
        id: "f2",
        q: "What is overfitting?",
        short: "The model has learned the noise in the training data rather than the underlying pattern — strong training performance, notably worse validation performance.",
        strong:
          "Concretely, the model has enough capacity to fit the idiosyncrasies of the specific rows it saw, so it does well on those and generalises poorly. I diagnose it from the gap between training and validation scores, and by watching the two curves during training: if training loss keeps falling while validation loss turns upward, that's the moment overfitting starts. The fixes depend on what's cheap — more data if I can get it, otherwise regularization, a simpler model, early stopping, or bagging. On neural networks, dropout and augmentation.",
        ref: "over-underfitting",
      },
      {
        id: "f3",
        q: "What is underfitting?",
        short: "The model is too simple to capture the pattern — poor training performance and poor validation performance, close together.",
        strong:
          "The signature is that both scores are bad and roughly equal, which means the model isn't even fitting the data it can see. That points at capacity or features rather than generalisation. I'd add capacity — a deeper tree, boosting instead of a single model, a kernel instead of a linear boundary — or, more often on tabular data, better features, because underfitting frequently means the signal is there but not in a form the model can express. I'd also reduce regularization and check the model has actually converged. And I'd sanity-check for a bug: persistent underfitting on a problem that should be easy usually means something is broken upstream.",
        ref: "over-underfitting",
      },
      {
        id: "f4",
        q: "Explain the bias–variance tradeoff.",
        short: "Bias is error from wrong assumptions — the model is too simple. Variance is error from sensitivity to the particular training sample. Reducing one typically increases the other.",
        strong:
          "Expected error decomposes into bias squared, variance, and irreducible noise. A simple model makes strong assumptions, so it's consistently wrong in the same way regardless of which sample it sees — high bias, low variance. A flexible model adapts closely to whatever data it got, so it's right on average but changes a lot between samples — low bias, high variance. As you increase complexity, bias falls and variance rises, and total error traces a U. Tuning is the search for the bottom of that U, and cross-validation is how I locate it empirically. The one thing you can't touch is the noise floor.",
        ref: "bias-variance",
      },
      {
        id: "f5",
        q: "What is data leakage?",
        short: "Information available during training that wouldn't be available at prediction time, producing offline scores that don't survive production.",
        strong:
          "It comes in two main forms. Target leakage is a feature that encodes the outcome — something populated only after the event you're predicting. Preprocessing leakage is fitting a transformation on the full dataset before splitting, so training rows are influenced by test rows. The tell is a metric that's too good, and the discipline is to put every fitted step inside a pipeline and to check every feature against the prediction timestamp. The strongest single test is evaluating on a genuinely later time period — leakage rarely survives that.",
        ref: "data-leakage",
      },
      {
        id: "f6",
        q: "Why do we need a validation set at all — why not just train and test?",
        short: "Because you need somewhere to make decisions that isn't the data you'll report your final number on.",
        strong:
          "Model selection and hyperparameter tuning are a search, and whatever data you score that search against becomes optimistically biased — you've partly selected for its noise. If the only held-out data is the test set, then the moment you use it to choose anything, you no longer have an unbiased estimate. The validation set absorbs that bias so the test set can stay clean. On small data I'd use cross-validation in place of a fixed validation split, but the principle is identical: the data used for decisions must be separate from the data used for the final number.",
        ref: "splits",
      },
      {
        id: "f7",
        q: "What's the difference between parametric and non-parametric models?",
        short: "Parametric models have a fixed number of parameters regardless of dataset size; non-parametric models grow in complexity with the data.",
        strong:
          "Linear regression is parametric — n features means n+1 coefficients whether you have a thousand rows or ten million. KNN is non-parametric in the extreme: the training set is the model, so it grows without bound. Decision trees are non-parametric too, since the tree gets deeper with more data. The practical implication is about assumptions and data hunger: parametric models impose a functional form, so they need less data but are wrong if the form is wrong; non-parametric models make fewer assumptions but need much more data and are more prone to overfitting.",
      },
      {
        id: "f8",
        q: "What is the curse of dimensionality?",
        short: "As dimensions grow, data becomes exponentially sparse and distances between points become uniformly similar, which breaks distance-based methods.",
        strong:
          "Two intertwined effects. Volume grows exponentially with dimension, so a fixed number of points covers a vanishingly small fraction of the space — you'd need exponentially more data to maintain the same density. And the ratio between the nearest and farthest neighbour distance converges to one, so 'nearest' stops carrying information. That's fatal for KNN, K-Means and RBF kernels. The mitigations are dimensionality reduction, feature selection, learned embeddings that put the structure into a lower-dimensional manifold, or using models that aren't distance-based — trees don't care nearly as much.",
      },
      {
        id: "f9",
        q: "What is the No Free Lunch theorem, in practical terms?",
        short: "Averaged over all possible problems, no algorithm beats any other. Which means every claim about a 'best model' is a claim about a class of problems.",
        strong:
          "The formal statement is about averaging over all possible target functions, which isn't a realistic setting — real problems have structure, so useful priors exist. The practical reading is: don't argue from authority about which algorithm is best, argue from the data and the constraints, and then measure. It's also a caution against transferring a benchmark result to a different domain. In an interview I'd use it to justify why I always fit a baseline and compare candidates on the same splits rather than reaching straight for the model that won the last competition.",
        ref: "model-selection",
      },
      {
        id: "f10",
        q: "What is regularization, in one sentence, and why does it work?",
        short: "A penalty on model complexity added to the training loss — it works by trading a little bias for a larger reduction in variance.",
        strong:
          "Without a penalty, the optimizer's only objective is to reduce training error, and the cheapest route to that is often large, finely-tuned weights that fit noise. Adding a penalty on the weights makes that expensive, so a coefficient only grows if the reduction in loss justifies it. That constrains the effective capacity of the model, which raises bias slightly and reduces variance substantially — a good trade whenever you're on the overfitting side of the curve. The same logic covers dropout, early stopping and tree depth limits, which are all regularization even though they don't look like a penalty term.",
        ref: "regularization",
      },
    ],
  },

  /* ================================================================ */
  {
    id: "qa-metrics",
    title: "Metrics",
    blurb: "Where interviews are won and lost. Every answer should end up connected to the cost of the two error types.",
    questions: [
      {
        id: "me1",
        q: "Precision vs recall — explain the difference and the tradeoff.",
        short: "Precision is TP/(TP+FP): of what I flagged, how much was real. Recall is TP/(TP+FN): of what was real, how much did I catch.",
        strong:
          "Precision's denominator is what you predicted positive; recall's denominator is what actually is positive. They trade off through the decision threshold: raise it and you flag fewer things, so precision usually goes up and recall necessarily goes down; lower it and the reverse. Which one to optimise is a business question — it depends on whether a false alarm or a miss costs more, and on what happens operationally after the model fires. If a flag triggers an automatic block, precision matters; if it lands in a human review queue, you can afford to push recall.",
        ref: "metrics",
      },
      {
        id: "me2",
        q: "When would you use F1 instead of accuracy?",
        short: "When classes are imbalanced and you care about the minority class, because accuracy is dominated by the majority class.",
        strong:
          "Accuracy weights every prediction equally, so with 99% negatives it's essentially a measurement of how well you predict negatives — and predicting 'negative' always scores 99%. F1 only involves TP, FP and FN, so true negatives don't enter it at all, which means the huge easy majority can't inflate it. I'd use F1 when I need one number, the classes are imbalanced, and I have no strong reason to prefer precision over recall. If I do have a reason, F-beta is more honest, and if I have several thresholds to consider, PR-AUC is better than any single-threshold metric.",
        ref: "metrics",
      },
      {
        id: "me3",
        q: "ROC-AUC vs PR-AUC — when does the choice matter?",
        short: "On imbalanced data. ROC-AUC's false-positive rate has an enormous denominator, so it stays flattering; PR-AUC uses precision and responds sharply to false positives.",
        strong:
          "ROC-AUC plots TPR against FPR = FP/(FP+TN). When negatives dominate, TN is huge, so thousands of false positives barely move FPR and the curve looks excellent. PR-AUC plots precision against recall, and precision's denominator is only the things you flagged, so it degrades immediately as false positives accumulate. Concretely: with 1,000 positives among a million rows, a model with 900 TP and 9,000 FP has a great ROC-AUC and precision of 9%. The caveat with PR-AUC is that its no-skill baseline is the positive rate rather than 0.5, so you always have to report the base rate alongside it.",
        ref: "metrics",
      },
      {
        id: "me4",
        q: "Why can accuracy be misleading?",
        short: "Because it's dominated by the majority class and treats both error types as equally costly.",
        strong:
          "Two independent problems. First, imbalance: at 1% positives, the trivial always-negative model scores 99%, so accuracy carries almost no information about the thing you care about. Always compare against the majority-class baseline before quoting accuracy. Second, asymmetric cost: even with balanced classes, accuracy implicitly says a false positive and a false negative are equally bad. In medical screening or fraud they are wildly different, and a model with better accuracy can easily be worse in expected cost. My default is to look at the confusion matrix rather than the single number.",
        ref: "metrics",
      },
      {
        id: "me5",
        q: "What does an AUC of 0.5 mean? What about 0.3?",
        short: "0.5 means the model ranks no better than random. 0.3 means it's ranking worse than random — usually a flipped label or inverted sign.",
        strong:
          "AUC is the probability that a randomly chosen positive scores above a randomly chosen negative. At 0.5 that's a coin flip and the model carries no ranking signal. Below 0.5 the ordering is systematically inverted, which is almost always a bug rather than a genuinely anti-predictive model — a swapped positive class label, a sign error, or predict_proba column indices mixed up. The giveaway is that 1 − AUC would be strong: an AUC of 0.3 means flipping the predictions gives 0.7, so the signal is there and pointing the wrong way.",
        ref: "metrics",
      },
      {
        id: "me6",
        q: "MAE vs RMSE — how do you choose?",
        short: "RMSE penalises large errors much more heavily; MAE treats all errors proportionally and is robust to outliers.",
        strong:
          "RMSE comes from squared error, so a single 10-unit miss contributes as much as a hundred 1-unit misses. That's the right behaviour when large errors are disproportionately damaging — an ETA that's two hours out is far worse than four that are thirty minutes out. MAE is the right choice when the data has outliers you don't want the model to chase, or when the business cost really is linear in the error. A useful diagnostic: compare them. If RMSE is much larger than MAE, your error distribution has a heavy tail and a few big misses dominate — that's worth investigating separately.",
        ref: "metrics",
      },
      {
        id: "me7",
        q: "What is model calibration and when do you care about it?",
        short: "A model is calibrated when its predicted probabilities match observed frequencies. You care whenever the probability is used as a number, not just as a ranking.",
        strong:
          "If a model says 0.7 for a thousand cases and roughly 700 of them are positive, it's calibrated. AUC can be perfect while calibration is terrible, because AUC only cares about ordering. Calibration matters when the probability feeds a decision — expected loss, pricing, a bid, or a threshold derived from business cost — because those calculations assume the number means something. I'd check with a reliability diagram plus Brier score or log loss, and fix with Platt scaling or isotonic regression fitted on held-out data. Worth knowing that class weighting and resampling both break calibration, so if you use them and need probabilities, recalibrate.",
        ref: "metrics",
      },
      {
        id: "me8",
        q: "How do you evaluate a multiclass classifier?",
        short: "Per-class precision and recall, then aggregate with macro, weighted or micro averaging — and say which you chose and why.",
        strong:
          "I'd start with the full confusion matrix, because it shows which classes get confused with which, and that's usually more actionable than any single number. Then per-class precision, recall and F1. For a headline figure, macro-averaging treats every class equally, so rare classes count as much as common ones — that's what I'd use if the rare classes matter. Weighted averaging weights by support, so it's closer to accuracy and lets rare classes get drowned out. Micro-averaging pools all the counts, and for single-label multiclass micro-F1 is just accuracy. Choosing the right one is a statement about what you care about, so I'd always say which and why.",
        ref: "metrics",
      },
      {
        id: "me9",
        q: "Your model has 0.92 AUC but the business says it's useless. What's going on?",
        short: "Almost certainly the operating point. Good ranking doesn't mean the threshold you deployed produces a usable precision or a usable volume.",
        strong:
          "AUC summarises the whole curve; the business experiences one point on it. Several things could be true. The classes may be so imbalanced that even a strong ranker gives 8% precision at the recall the business needs, so the review queue is mostly noise — PR-AUC would have shown that. Or the threshold might be flagging far more or far less volume than the team can act on, in which case precision@K against their actual capacity is the metric that matters. Or the model may be strong on the easy majority and weak on the segment they care about, which slicing would reveal. I'd go back to their operational constraint — how many cases can you action per day, and what does a wrong one cost — and re-derive the threshold from that.",
        ref: "metrics",
      },
    ],
  },

  /* ================================================================ */
  {
    id: "qa-training",
    title: "Model Training",
    blurb: "Optimization, regularization and the mechanics of getting a model to converge.",
    questions: [
      {
        id: "t1",
        q: "What happens if the learning rate is too high?",
        short: "The updates overshoot the minimum. The loss oscillates, spikes, or diverges to NaN.",
        strong:
          "Each step moves further than the curvature of the loss justifies, so instead of descending you bounce across the valley, sometimes landing further from the minimum than you started. On a quadratic loss with a step size above a specific threshold, the error grows every step and the loss diverges. In practice you see the loss go flat-then-spike, or become NaN once the numbers overflow. The fixes are to reduce the learning rate by a factor of ten, add gradient clipping, use warmup so early unstable steps are small, and make sure the inputs are scaled — unscaled features create an ill-conditioned surface where no single learning rate works for all parameters.",
        ref: "gradient-descent",
      },
      {
        id: "t2",
        q: "Why do we use regularization?",
        short: "To reduce overfitting by penalising complexity, trading a little bias for a larger reduction in variance.",
        strong:
          "Unconstrained optimisation drives training error down by whatever means available, which usually means large weights that fit noise. A penalty makes weight magnitude costly, so the model has to justify each coefficient with a real reduction in loss. That shrinks the effective hypothesis space, which raises bias slightly and cuts variance substantially. I choose the strength by cross-validating over a log-spaced grid, and I always scale the features first — otherwise the penalty is applied unevenly based on arbitrary units.",
        ref: "regularization",
      },
      {
        id: "t3",
        q: "L1 vs L2 — the short version.",
        short: "L1 penalises absolute weights and drives some exactly to zero, giving feature selection. L2 penalises squared weights and shrinks everything smoothly without eliminating anything.",
        strong:
          "The mechanism is the gradient of the penalty. L2's gradient is 2w, which shrinks as w shrinks, so the pull toward zero weakens near zero and never quite arrives. L1's gradient is a constant ±λ, so it keeps pushing at full strength and small weights land exactly on zero. Practically: L1 when I expect a sparse solution or want built-in feature selection; L2 when I expect many small effects, especially with correlated features, because L1 arbitrarily picks one of a correlated group and zeroes the rest. Elastic Net is the standard compromise for wide, correlated data.",
        ref: "regularization",
      },
      {
        id: "t4",
        q: "Batch, stochastic and mini-batch gradient descent — what's the difference?",
        short: "How many rows are used per parameter update: all of them, one, or a small batch.",
        strong:
          "Full-batch computes the exact gradient over the whole dataset, so descent is smooth and stable, but you get one update per pass and it's infeasible if the data doesn't fit in memory. Stochastic uses a single row per update, which is very fast per step and the noise can help escape saddle points, but the trajectory is erratic and you lose vectorisation. Mini-batch — typically 32 to 512 rows — is the practical default: a good gradient estimate, full hardware utilisation, and enough gradient noise to help generalisation. In modern usage 'SGD' almost always means mini-batch SGD.",
        ref: "gradient-descent",
      },
      {
        id: "t5",
        q: "What's the difference between an epoch, a batch and an iteration?",
        short: "An epoch is one full pass over the training data. A batch is the group of rows in one update. An iteration is one parameter update.",
        strong:
          "With 10,000 rows and a batch size of 100, one epoch is 100 iterations. It matters because papers and libraries report schedules in different units — a learning rate decay 'every 1000 steps' means something very different at batch size 32 versus 512. It also matters when comparing training runs: two runs with the same epoch count but different batch sizes have done very different numbers of updates.",
        ref: "gradient-descent",
      },
      {
        id: "t6",
        q: "What is early stopping, and is it a form of regularization?",
        short: "Halting training when validation performance stops improving. Yes — it's regularization, because it limits the effective capacity the model gets to use.",
        strong:
          "You monitor a validation metric each epoch or each boosting round, and stop when it hasn't improved for a set patience, restoring the best checkpoint. It's regularization in the meaningful sense: it constrains how far the optimizer travels from initialisation, and for linear models with gradient descent early stopping is provably related to L2 regularization. It's also the cheapest form there is, because you get it as a byproduct of training. The one requirement is a validation set that isn't your test set — the stopping point is a decision, and decisions cost you the cleanliness of whatever data informed them.",
        ref: "over-underfitting",
      },
      {
        id: "t7",
        q: "Grid search vs random search — which and why?",
        short: "Random search, usually. At an equal budget it explores each hyperparameter's range much more finely, and only a few hyperparameters typically matter.",
        strong:
          "With a grid, every trial repeats the same values of every hyperparameter — nine grid points over two hyperparameters gives only three distinct values of each. Random search with nine trials gives nine distinct values of each. Since in most problems only one or two hyperparameters have a large effect and you don't know which in advance, random search gets much better resolution on whichever one matters. Grid search is still fine for one or two hyperparameters with a small, well-understood range. For expensive training runs I'd go further and use Bayesian optimization, which models the objective and picks promising points, or Hyperband, which kills bad configurations early.",
        ref: "params-hyperparams",
      },
      {
        id: "t8",
        q: "Your training loss is decreasing but validation loss has started rising. What do you do?",
        short: "That's the onset of overfitting. Stop at the validation minimum, then add regularization or data.",
        strong:
          "The crossover point is exactly what early stopping is for, so first I'd restore the checkpoint at the validation minimum rather than the final one. Then I'd address the underlying capacity mismatch: more training data if it's obtainable, stronger regularization — higher weight decay, more dropout, shallower trees — or a smaller model. On images or text I'd add augmentation. I'd also check that the validation set is large enough to be trustworthy; a rising curve on 200 validation rows might just be noise.",
        ref: "over-underfitting",
      },
      {
        id: "t9",
        q: "Why do we subtract the gradient rather than add it?",
        short: "Because the gradient points in the direction of steepest increase of the loss, and we want to decrease the loss.",
        strong:
          "The gradient of J with respect to θ is the vector of partial derivatives, and it points uphill on the loss surface — the direction in which the loss grows fastest. Since the objective is minimisation, we move in the opposite direction, hence θ ← θ − η∇J. If you were maximising something instead — a reward, or a likelihood you'd chosen not to negate — you'd add it, and that's gradient ascent. It's a small point but people do get asked it, and stumbling on it is a bad look.",
        ref: "gradient-descent",
      },
    ],
  },

  /* ================================================================ */
  {
    id: "qa-algorithms",
    title: "Algorithms",
    blurb: "Comparisons and mechanism questions. Interviewers are checking whether you understand why, not just what.",
    questions: [
      {
        id: "a1",
        q: "Random forest vs gradient boosting?",
        short: "Random forest builds deep trees independently in parallel and averages them to reduce variance. Boosting builds shallow trees sequentially, each correcting the ensemble's errors, to reduce bias.",
        strong:
          "Independence is the structural difference. A random forest trains each tree on its own bootstrap sample with random feature subsets, so training parallelises perfectly, the trees are deliberately deep, and averaging kills the variance. Boosting trains each tree on the residual errors of everything so far, so it's sequential, and the trees are deliberately shallow because the ensemble's bias falls with each round. The consequences: forests are hard to overfit and need almost no tuning, so they're an excellent first model; GBMs usually win on accuracy but need early stopping and careful tuning of learning rate against tree count. With no tuning budget, I'd take the forest. With a budget, LightGBM.",
        ref: "algorithms",
      },
      {
        id: "a2",
        q: "Why don't tree-based models require feature scaling?",
        short: "Splits are threshold comparisons on one feature at a time, and any monotone rescaling preserves the ordering, so the same rows go each way.",
        strong:
          "A split asks 'is income below 50,000'. Standardize the column and the equivalent split is 'is income_z below 0.38' — identical partition, identical impurity gain, identical tree. Because the algorithm only ever compares values within a single feature, and never combines features into a distance or a weighted sum, the units are irrelevant. The corollary is that log transforms don't help trees either, for the same reason — whereas they can substantially help a linear model. Contrast that with KNN, SVM, PCA and neural networks, all of which combine features and therefore depend on their relative scales.",
        ref: "feature-scaling",
      },
      {
        id: "a3",
        q: "Logistic regression vs SVM?",
        short: "Similar linear boundaries, different losses. Logistic regression gives probabilities; SVM maximises the margin and gives an uncalibrated score.",
        strong:
          "Logistic regression uses log loss, which never fully stops caring — even correctly classified points far from the boundary contribute a small gradient. SVM uses hinge loss, which is exactly zero beyond the margin, so only the support vectors influence the fit. That makes SVM slightly more robust to well-classified outliers and gives it a sparse solution. But SVM's output is a distance, not a probability, so you'd need Platt scaling if you want calibrated outputs, and it scales badly beyond around 100,000 rows. My default is logistic regression for the probabilities and speed; I'd reach for an SVM with an RBF kernel on small, high-dimensional, clean data where the boundary is nonlinear.",
        ref: "algorithms",
      },
      {
        id: "a4",
        q: "What are the advantages and disadvantages of KNN?",
        short: "No training, naturally nonlinear, easy to explain. But inference is expensive, it needs scaling, and it collapses in high dimensions.",
        strong:
          "The appeal is simplicity: no training phase, arbitrary decision boundaries, and a prediction you can justify by pointing at the neighbours. The costs are real though. Inference is O(N·d) per query with a naive scan and you must keep the whole training set in memory, so it doesn't scale to high query volumes. Scaling is mandatory because everything is a distance. And in high dimensions distances concentrate, so 'nearest' stops being meaningful. It's also sensitive to irrelevant features, since every useless dimension adds noise to every distance. That said, the idea is very much alive — approximate nearest-neighbour search over embeddings is exactly this, and it's the retrieval layer in every RAG system.",
        ref: "algorithms",
      },
      {
        id: "a5",
        q: "Why is Naive Bayes called naive?",
        short: "Because it assumes every feature is conditionally independent of every other given the class — an assumption that's essentially never true.",
        strong:
          "In text classification, 'free' and 'money' clearly co-occur, so treating them as independent double-counts their evidence and makes the model overconfident — Naive Bayes routinely outputs probabilities like 0.9999. But classification only needs the argmax to be right, not the probabilities, and the distortion tends to affect competing classes similarly, so the ranking often survives. That's why it remains a strong, extremely fast text baseline despite an assumption that's obviously false. The practical rule: use its predictions, don't use its probabilities.",
        ref: "algorithms",
      },
      {
        id: "a6",
        q: "How does a decision tree handle a continuous feature?",
        short: "It sorts the unique values and evaluates candidate thresholds between them, picking the split with the best impurity reduction.",
        strong:
          "For each numeric feature the algorithm sorts the values present at that node and considers thresholds at the midpoints between consecutive distinct values. For each candidate it computes the weighted impurity of the two children and takes the best. That's why tree training is dominated by sorting, and why histogram-based implementations like LightGBM are so much faster — they bucket continuous values into a few hundred bins up front and only consider bin boundaries, which is a huge reduction in candidate splits for almost no loss in quality.",
        ref: "algorithms",
      },
      {
        id: "a7",
        q: "When would you use PCA before training, and when would you not?",
        short: "Use it when you have many correlated features, need speed, or want to visualise. Avoid it when interpretability matters or when you're using a tree ensemble.",
        strong:
          "PCA helps when p is large relative to n, when features are heavily correlated and the model dislikes multicollinearity, when training cost is a bottleneck, or when you want a 2D picture. It hurts in a few specific ways. It destroys interpretability, since every component is a blend of everything. It's unsupervised, so a low-variance direction that happens to predict y perfectly can be discarded — there's nothing in the objective that knows about the target. And it's usually unhelpful before a gradient boosting model, which handles correlated features fine and loses its ability to make clean axis-aligned splits on meaningful features once you've rotated the space.",
        ref: "dimensionality-reduction",
      },
      {
        id: "a8",
        q: "Explain the kernel trick.",
        short: "Compute inner products as if the data had been mapped into a much higher-dimensional space, without ever performing the mapping.",
        strong:
          "The SVM's dual formulation only ever needs inner products between pairs of points, never the points themselves. A kernel function K(x, x') returns the inner product that would result from applying some feature map to both points, so you get the benefit of a high-dimensional — for the RBF kernel, infinite-dimensional — representation at the cost of a single scalar evaluation. That's what makes a nonlinear boundary tractable. The parameter to know is gamma in the RBF kernel: high gamma means each point's influence decays quickly, producing a tight wiggly boundary that overfits; low gamma means broad influence and a nearly linear boundary that underfits.",
        ref: "algorithms",
      },
      {
        id: "a9",
        q: "You have 1,000 rows and 5,000 features. What do you use?",
        short: "A strongly regularized linear model or an SVM, plus aggressive feature reduction. Not a deep network, and probably not a big GBM.",
        strong:
          "With p far larger than n, almost anything flexible will separate the training data perfectly and generalise terribly. I'd start with L1 or Elastic Net logistic regression, which handles wide data directly and gives me a sparse, inspectable solution. A linear SVM is the other natural choice — the maximum-margin objective is well-suited to this regime. I'd also consider univariate filtering or PCA first to cut dimensionality, and I'd use repeated cross-validation rather than a single split, because at 1,000 rows any single estimate is noisy. And I'd be sceptical of anything that looks too good — with 5,000 features, spurious correlations with the target are guaranteed.",
        ref: "model-selection",
      },
    ],
  },

  /* ================================================================ */
  {
    id: "qa-practical",
    title: "Practical ML",
    blurb: "The debugging and judgement questions. These are the ones that reveal whether you have actually shipped something.",
    questions: [
      {
        id: "p1",
        q: "What would you do with an imbalanced dataset?",
        short: "Change the metric, tune the threshold, add class weights, then — only if needed — resample inside the training fold.",
        strong:
          "In that order, because the cheap fixes usually do most of the work. First the metric: accuracy is out, PR-AUC and precision/recall at a chosen operating point are in, and splits should be stratified. Then the threshold: 0.5 is a convention, and moving it costs nothing and requires no retraining. Then class weights, which is one argument and keeps the true distribution. Only then would I try resampling like SMOTE, strictly inside the training folds, and I'd expect it to help less than its reputation suggests. Finally, if the probabilities feed a downstream calculation, I'd recalibrate, because weighting and resampling both distort them.",
        ref: "class-imbalance",
      },
      {
        id: "p2",
        q: "How would you detect overfitting?",
        short: "Compare training and validation performance. A large and growing gap is the signature.",
        strong:
          "The primary check is the gap between training and validation scores — a model at 0.99 train and 0.74 validation is overfitting regardless of what else is true. Beyond the single number I'd look at learning curves over training: if training loss keeps improving while validation loss turns up, that's the crossover. I'd also plot performance against training set size — if validation performance is still climbing as I add data, more data will help, which is a variance problem. And I'd check the cross-validation spread, since high variance across folds is itself a symptom. What I wouldn't do is judge from training performance alone, which by construction cannot reveal it.",
        ref: "over-underfitting",
      },
      {
        id: "p3",
        q: "Training accuracy is high but validation accuracy is low. What do you do?",
        short: "Overfitting. Add regularization, simplify the model, get more data, or apply early stopping — and check the split for leakage or mismatch first.",
        strong:
          "First I'd rule out an infrastructure explanation: is the validation set drawn from the same distribution, is there a time or group structure the split ignored, is the validation set big enough to trust. Assuming the split is sound, this is textbook high variance. My ordered list: more training data if obtainable, since it's the most reliable fix; then stronger regularization — raise λ, reduce max_depth, increase min_samples_leaf, add dropout; then a simpler model family; then early stopping; then augmentation if the data type allows. I'd also consider reducing the feature count, because a large feature set with limited rows is a common cause.",
        ref: "over-underfitting",
      },
      {
        id: "p4",
        q: "Both training and validation accuracy are low. What now?",
        short: "Underfitting. Add capacity, improve the features, train longer, reduce regularization — and check for a bug.",
        strong:
          "Both being low and close means the model isn't even fitting data it can see, so generalisation isn't the issue — expressiveness is. I'd increase capacity, move from a linear model to a gradient boosting model, or add an RBF kernel. I'd reduce regularization. I'd check that training actually converged rather than stopping early or stalling at a bad learning rate. But on tabular data the biggest lever is usually features: engineer interactions, ratios and domain-derived signals, because underfitting often means the signal exists but not in a form the model can express. And I'd sanity-check for bugs — a misaligned target, a scrambled join, or leakage of noise into the features all look like underfitting.",
        ref: "over-underfitting",
      },
      {
        id: "p5",
        q: "How do you handle missing values?",
        short: "Understand why they're missing first, then choose: drop, impute, or flag — and add a missing indicator when the absence itself is informative.",
        strong:
          "I'd start by quantifying it — how much, in which columns, and whether missingness correlates with anything, including the target. Then I'd ask whether it's plausibly random or systematic, because that changes the right answer. For most tabular work I use median imputation for numerics and an explicit 'Missing' category for categoricals, and I add a binary missing-indicator whenever the missingness looks informative — which is often, and it's cheap insurance. If I'm using a gradient boosting library I'd often just let it handle missing values natively, since it learns a default direction at each split. All of it goes inside a pipeline so the statistics come from the training fold only.",
        ref: "missing-data",
      },
      {
        id: "p6",
        q: "How do you handle categorical features?",
        short: "One-hot for low-cardinality unordered categories, ordinal for genuinely ordered ones, and out-of-fold target encoding or embeddings for high cardinality.",
        strong:
          "The encoding is a claim about the categories, so I pick the claim that's true. Unordered with few levels gets one-hot. Genuinely ordered levels — small, medium, large — get ordinal encoding, which preserves the order. What I never do is assign arbitrary integers to unordered categories for a linear or distance-based model, because that invents an ordering and a spacing that don't exist. For high cardinality like zip code or merchant id, I'd use frequency encoding as a safe baseline or out-of-fold target encoding with smoothing for a GBM, or let CatBoost handle it. And I'd always define what happens when an unseen category arrives in production.",
        ref: "categorical",
      },
      {
        id: "p7",
        q: "Your model performed well offline but is failing in production. Where do you look?",
        short: "Leakage, train/serve skew, distribution shift, or a threshold chosen without regard to the real operating conditions.",
        strong:
          "Four candidates, in the order I'd check them. Leakage is first: a feature that was available offline but is empty or late at serving time will silently gut the model. Train/serve skew is second: the offline feature pipeline and the online one computing the same feature differently — different null handling, different time window, different rounding. Third, distribution shift: the production population may simply not match the training population, especially if the training data came from a period with different conditions. Fourth, the operating point: a threshold tuned for a metric nobody actually cares about. I'd start by logging the actual feature vectors used at inference and comparing their distributions to training — that finds skew and shift quickly, and it's the cheapest diagnostic.",
        ref: "pipeline",
      },
      {
        id: "p8",
        q: "How would you explain a model's prediction to a non-technical stakeholder?",
        short: "Use global feature importance for the model overall and SHAP for individual predictions, translated into their language.",
        strong:
          "Two levels. Globally, permutation importance answers 'what does the model pay attention to' — and I'd use permutation rather than the default impurity importance, which is biased toward high-cardinality features. For a single decision, SHAP values decompose the prediction into per-feature contributions relative to the average, which maps naturally onto 'this application scored high mainly because of X and Y, partly offset by Z'. I'd translate into their units rather than showing coefficients, and I'd be explicit that these are associations the model learned, not causal claims — that distinction matters a lot when someone is about to act on it.",
      },
      {
        id: "p9",
        q: "You've been given a dataset and a target. What are the first things you do?",
        short: "Understand the target and the timing, look at the data, establish a baseline, and set up a split you trust — before any modelling.",
        strong:
          "In order: I'd confirm exactly what the target means and when it becomes known, because that determines everything about leakage. Then EDA — row counts, class balance, missingness, duplicates, distributions, obvious data errors, and whether there's a time or group structure. Then I'd design the split accordingly: stratified, grouped, or time-based. Then a baseline — majority class or a simple logistic regression — which tells me how hard the problem is and catches broken data early. Only then would I start modelling. The order matters: most of the expensive mistakes in an ML project are made before the first model is trained.",
        ref: "pipeline",
      },
      {
        id: "p10",
        q: "How much data do you need?",
        short: "It depends on the signal-to-noise ratio and model complexity — and the empirical way to find out is a learning curve.",
        strong:
          "There's no universal number, but there is a way to answer it empirically: train on 10%, 25%, 50% and 100% of the data and plot validation performance against training size. If the curve is still rising at 100%, more data will help and that's the cheapest available improvement. If it's flat, more of the same data won't help and you need better features, a different model, or better labels. That's a much more useful answer than a rule of thumb, and it also tells you whether you're in a bias-limited or variance-limited regime, which determines what to do next.",
      },
    ],
  },
];

export const TOTAL_QUESTIONS = QA_GROUPS.reduce((n, g) => n + g.questions.length, 0);
