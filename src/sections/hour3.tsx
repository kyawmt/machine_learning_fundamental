import { Callout } from "../components/Callout";
import { DefinitionCard } from "../components/DefinitionCard";
import { ConceptCard } from "../components/ConceptCard";
import { InterviewQuestion } from "../components/InterviewQuestion";
import {
  Chip,
  Code,
  ComparisonTable,
  Formula,
  FlowDiagram,
  Grid,
  Panel,
  SubHeading,
} from "../components/ui";
import { GradientDescentLab } from "../components/viz/GradientDescentLab";
import { RegularizationLab } from "../components/viz/RegularizationLab";

/* ================================================================== */
/* 14 · Regularization                                                 */
/* ================================================================== */

export function Regularization() {
  return (
    <>
      <ConceptCard
        title="Regularization"
        priority="must"
        whatItIs="Adding a penalty on model complexity to the training objective, so the optimizer has to trade fit against simplicity instead of chasing the training data as far as it can."
        intuition="Without a penalty, the optimizer's only instruction is 'reduce training error', and the easiest way to do that is to grow large, finely-tuned weights that latch onto noise. The penalty makes those weights expensive, so a coefficient only gets large if it genuinely pays for itself."
        example="Predicting house price from 200 features with 500 rows. Unregularized least squares fits the noise and produces wild coefficients. Ridge shrinks them all toward zero; the fit is slightly worse on training data and substantially better on held-out data."
        keyTakeaway="Regularization deliberately increases bias to reduce variance. It's the standard treatment for overfitting."
      >
        <Formula caption="The penalty is added to the training loss. λ (alpha in scikit-learn) controls how much it matters.">
          {`Objective  =  Loss(data)  +  λ · Penalty(weights)
                    ↑                    ↑
              fit the data        keep weights small`}
        </Formula>
      </ConceptCard>

      <RegularizationLab />

      <Grid cols={2}>
        <ConceptCard
          title="L1 — Lasso"
          priority="must"
          whatItIs={<>Penalty is the sum of absolute weights: <Code>Loss + λ Σ|wⱼ|</Code></>}
          intuition="The gradient of |w| is a constant ±λ regardless of how small w is, so the penalty keeps pushing with full force right up to zero — and weights land exactly on zero and stay there. Geometrically, the constraint region is a diamond, and diamonds have corners on the axes."
        >
          <ul className="bullet-list">
            <li>Produces <strong>sparse</strong> models — many coefficients are exactly 0.</li>
            <li>Performs <strong>automatic feature selection</strong>: the zeroed features are dropped.</li>
            <li>Among correlated features it tends to keep one arbitrarily and zero the rest — unstable if you care about which.</li>
            <li>No closed-form solution; solved by coordinate descent or subgradient methods.</li>
            <li>Great when you believe only a handful of features actually matter.</li>
          </ul>
        </ConceptCard>

        <ConceptCard
          title="L2 — Ridge"
          priority="must"
          whatItIs={<>Penalty is the sum of squared weights: <Code>Loss + λ Σwⱼ²</Code></>}
          intuition="The gradient of w² is 2w, which shrinks as w shrinks — so the push toward zero weakens as you approach it and never quite arrives. The constraint region is a smooth circle with no corners to land on."
        >
          <ul className="bullet-list">
            <li><strong>Shrinks</strong> all coefficients toward zero, essentially never to exactly zero.</li>
            <li>Handles correlated features gracefully — it <strong>spreads weight across them</strong> instead of picking one.</li>
            <li>Has a closed-form solution and is numerically well behaved.</li>
            <li>In neural networks this is <strong>weight decay</strong>.</li>
            <li>The safer default when you think most features carry a little signal.</li>
          </ul>
        </ConceptCard>
      </Grid>

      <ComparisonTable
        headers={["", "L1 (Lasso)", "L2 (Ridge)", "Elastic Net"]}
        rows={[
          ["Penalty", "λ Σ|w|", "λ Σw²", "λ(α Σ|w| + (1−α) Σw²)"],
          ["Weights hit exactly zero", "Yes", "No (asymptotically only)", "Yes"],
          ["Feature selection", "Built in", "None", "Built in"],
          ["Correlated features", "Picks one arbitrarily", "Spreads weight across the group", "Selects groups together"],
          ["Solution", "No closed form", "Closed form", "No closed form"],
          ["Best when", "You expect few features to matter", "You expect many small effects", "High dimensional and correlated (p ≫ n, genomics, text)"],
          ["scikit-learn", "Lasso, LogisticRegression(penalty='l1')", "Ridge, LogisticRegression(penalty='l2')", "ElasticNet, penalty='elasticnet'"],
        ]}
      />

      <Callout variant="memorize" title="What λ does">
        <div className="grid gap-3 sm:grid-cols-2">
          <Formula>{`λ = 0        no penalty
             → plain least squares
             → low bias, high variance
             → overfits

λ small      light shrinkage

λ large      heavy shrinkage
             → high bias, low variance
             → underfits

λ → ∞        all weights → 0
             → predicts the intercept`}</Formula>
          <div>
            <p className="mb-2 text-[0.9rem] leading-relaxed text-muted">
              Increasing λ moves you <strong>left</strong> along the complexity axis of the bias–variance curve:
              bias up, variance down. There is an optimum in the middle, and you find it by cross-validation —{" "}
              <Code>RidgeCV</Code>, <Code>LassoCV</Code>, or a grid over{" "}
              <Code>C = 1/λ</Code> for logistic regression and SVMs.
            </p>
            <Callout variant="mistake" title={null}>
              <strong>Scale your features before regularizing.</strong> The penalty applies equally to every
              coefficient, so a feature measured in small units needs a large coefficient and gets punished for it —
              purely because of its unit. Also: don't penalise the intercept.
            </Callout>
          </div>
        </div>
      </Callout>

      <Callout variant="tip" title="Regularization is broader than L1 and L2">
        Anything that constrains capacity to improve generalisation counts, and naming a few earns credit:{" "}
        <strong>dropout</strong> (randomly zero activations), <strong>early stopping</strong> (stop before the model
        fits the noise), <strong>data augmentation</strong> (more effective variety), <strong>max_depth</strong> and{" "}
        <Code>min_samples_leaf</Code> on trees, <strong>subsample</strong> / <Code>colsample_bytree</Code> in GBMs,{" "}
        <strong>batch normalisation</strong> (partly), and <strong>label smoothing</strong>.
      </Callout>

      <SubHeading>Interview questions</SubHeading>
      <div className="space-y-2">
        <InterviewQuestion
          qa={{
            id: "rg1",
            q: "L1 vs L2 — what's the difference and when do you use each?",
            short: "L1 penalises absolute weight and drives some coefficients exactly to zero, so it selects features. L2 penalises squared weight and shrinks everything smoothly without eliminating anything.",
            strong:
              "The behavioural difference comes from the gradient. For L2 the penalty gradient is 2w, so the push toward zero fades as the weight shrinks and it converges toward zero without reaching it. For L1 the gradient is a constant ±λ, so the push doesn't weaken and small weights get driven exactly to zero and pinned there. Practically: I use L1 when I have many features and believe only a few matter, or when I want a sparse, cheap-to-serve model. I use L2 when I think most features contribute a little, especially with correlated features — L1 will arbitrarily keep one of a correlated group and zero the others, which makes the selected set unstable across resamples, whereas L2 distributes weight across them. Elastic Net combines both and is the standard answer for wide, correlated data.",
            deeper:
              "The geometric picture: minimising loss subject to a constraint on the weights. L2's constraint region is a sphere; L1's is a diamond with corners on the axes. The loss contours are ellipses expanding until they touch the region, and they're far more likely to touch a diamond at a corner — where one coordinate is exactly zero — than to touch a smooth sphere at an axis point.",
          }}
        />
        <InterviewQuestion
          qa={{
            id: "rg2",
            q: "What happens to bias and variance as you increase λ?",
            short: "Bias goes up, variance goes down. Push it far enough and the model underfits.",
            strong:
              "A larger penalty constrains the weights harder, so the model has less freedom to adapt to whatever training sample it got — that's variance falling. The same constraint stops it representing the true function fully, which is bias rising. Total error follows a U: it improves as you damp out noise-fitting, then degrades once you've suppressed real signal. I find the bottom of the U by cross-validating over a log-spaced λ grid, and I'd look at the whole validation curve rather than just the argmin, because a flat region means the choice isn't sensitive.",
          }}
        />
        <InterviewQuestion
          qa={{
            id: "rg3",
            q: "Why do you have to scale features before applying regularization?",
            short: "Because the penalty is applied uniformly to all coefficients, so features with small units get unfairly penalised.",
            strong:
              "Take income in dollars and income in thousands of dollars — the same information, but the coefficient in the thousands version has to be 1000× larger to produce the same prediction. A squared penalty then charges that coefficient a million times more. So the regularizer's behaviour depends on your arbitrary choice of unit, which is clearly wrong. Standardizing puts every feature on the same footing so the penalty reflects actual importance rather than measurement scale. This is also why the intercept is excluded from the penalty — it just centres the predictions and shouldn't be shrunk.",
          }}
        />
      </div>
    </>
  );
}

/* ================================================================== */
/* 15 · Gradient Descent                                               */
/* ================================================================== */

export function GradientDescent() {
  return (
    <>
      <ConceptCard
        title="Gradient Descent"
        priority="must"
        whatItIs="An iterative optimization method: repeatedly compute the gradient of the loss with respect to the parameters, and take a small step in the opposite direction."
        intuition="You're on a hillside in fog. You can't see the valley, but you can feel which way the ground slopes. Take a step downhill, feel again, repeat. The learning rate is how big a step you take."
        keyTakeaway="θ ← θ − η∇J(θ). The gradient points uphill, so we subtract it. η controls step size, not direction."
      >
        <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          <FlowDiagram
            steps={[
              { label: "1. Initialize parameters", note: "random, or zeros for a convex problem", tone: "accent" },
              { label: "2. Forward pass — predict" },
              { label: "3. Compute the loss" },
              { label: "4. Compute gradients ∇J(θ)", note: "backpropagation in a neural net" },
              { label: "5. Update: θ ← θ − η∇J(θ)", tone: "accent" },
              { label: "6. Repeat until converged", note: "loss plateaus, or early stopping fires", tone: "green" },
            ]}
            compact
          />
          <div className="space-y-3">
            <Formula caption="The single most important formula in machine learning training.">
              {`θ  ←  θ  −  η · ∇J(θ)

θ      model parameters (weights)
η      learning rate — the step size
∇J(θ)  gradient of the loss w.r.t. θ`}
            </Formula>
            <DefinitionCard term="Why minus?">
              The gradient <Code>∇J(θ)</Code> points in the direction of <strong>steepest increase</strong> of the
              loss. We want the loss to go <em>down</em>, so we move in the opposite direction. That's the entire
              reason for the minus sign — and it's a question people genuinely get asked.
            </DefinitionCard>
          </div>
        </div>
      </ConceptCard>

      <GradientDescentLab />

      <SubHeading priority="must">Batch, Stochastic, Mini-batch</SubHeading>
      <ComparisonTable
        headers={["Variant", "Rows per update", "Pros", "Cons"]}
        rows={[
          [
            "Batch (full-batch)",
            "All N",
            "Exact gradient; smooth, stable, monotone descent on a convex loss.",
            "One update per pass over the data. Infeasible if the data doesn't fit in memory. Slow.",
          ],
          [
            "Stochastic (SGD)",
            "1",
            "Very frequent updates; the noise can knock you out of poor local minima and saddle points.",
            "Extremely noisy path; no vectorisation benefit; usually needs a decaying learning rate to settle.",
          ],
          [
            "Mini-batch",
            "32 – 512 typically",
            "The practical default. Good gradient estimate, full GPU utilisation, and enough noise to help generalisation.",
            "Batch size becomes another hyperparameter, and it interacts with the learning rate.",
          ],
        ]}
        caption="In modern practice 'SGD' almost always means mini-batch SGD. Nobody trains a neural network one sample at a time."
      />

      <SubHeading priority="must">Learning rate failure modes</SubHeading>
      <Grid cols={3}>
        <Panel title="η too high" tone="red">
          <ul className="bullet-list">
            <li>Loss oscillates, spikes, or goes to NaN.</li>
            <li>Each step overshoots the minimum and lands further away than it started.</li>
            <li>Fix: divide by 10 and retry. Add gradient clipping. Use warmup.</li>
          </ul>
        </Panel>
        <Panel title="η too low" tone="neutral">
          <ul className="bullet-list">
            <li>Loss falls, but agonisingly slowly — often looks like a flat line.</li>
            <li>Wastes compute, and can stall in a plateau before reaching a good region.</li>
            <li>Fix: increase by 3–10×. Run an LR range test.</li>
          </ul>
        </Panel>
        <Panel title="η about right" tone="green">
          <ul className="bullet-list">
            <li>Loss falls quickly at first, then flattens smoothly.</li>
            <li>Small bumps are fine and expected with mini-batches.</li>
            <li>Decay it over training — cosine or step schedules — to settle into the minimum.</li>
          </ul>
        </Panel>
      </Grid>

      <Grid cols={2}>
        <DefinitionCard term="The optimizers you should be able to name">
          <ul className="bullet-list">
            <li>
              <strong>SGD + momentum.</strong> Accumulates a velocity term, so it accelerates along consistent
              directions and damps oscillation across ravines.
            </li>
            <li>
              <strong>RMSProp.</strong> Per-parameter learning rates scaled by a running average of squared
              gradients — parameters with large gradients get smaller steps.
            </li>
            <li>
              <strong>Adam.</strong> Momentum plus RMSProp, with bias correction. The default for most deep learning.
            </li>
            <li>
              <strong>AdamW.</strong> Adam with weight decay decoupled from the gradient update. The current default
              for transformers, and the right answer if someone asks what you'd use to train one.
            </li>
          </ul>
        </DefinitionCard>
        <Callout variant="mistake" title="Things people get wrong here">
          <ul className="bullet-list">
            <li>
              <strong>"Gradient descent finds the global minimum."</strong> Only for convex losses — linear
              regression, logistic regression, linear SVM. Neural network losses are non-convex, and in practice
              you're looking for a good-enough minimum, not the best one.
            </li>
            <li>
              <strong>"Local minima are the big problem in deep learning."</strong> In high dimensions, saddle points
              and flat plateaus are the more common obstacle; most local minima found are of comparable quality.
            </li>
            <li>
              <strong>Forgetting to scale inputs.</strong> Unscaled features create a stretched, ill-conditioned loss
              surface where no single learning rate works for all parameters.
            </li>
            <li>
              <strong>Confusing an epoch with a step.</strong> One epoch is a full pass over the data; one step is one
              parameter update. With batch size 64 and 6,400 rows, an epoch is 100 steps.
            </li>
          </ul>
        </Callout>
      </Grid>
    </>
  );
}

/* ================================================================== */
/* 16 · Parameters vs Hyperparameters                                  */
/* ================================================================== */

export function ParamsHyperparams() {
  return (
    <>
      <ConceptCard
        title="Parameters vs Hyperparameters"
        priority="must"
        whatItIs="Parameters are learned from data by the training algorithm. Hyperparameters are set by you before training and control how that learning happens."
        intuition="Hyperparameters are the dials on the machine; parameters are what the machine produces. Training optimises parameters. You (or a search) optimise hyperparameters, using validation performance as the signal."
        interviewAnswer="Parameters are fitted by the optimizer — the coefficients in a linear model, the weights in a network, the split thresholds in a tree. Hyperparameters are chosen before training and shape the process itself: learning rate, tree depth, number of estimators, regularization strength, batch size. The clean test is: was this value produced by gradient descent or by me?"
        keyTakeaway="Learned from data = parameter. Chosen by you before training = hyperparameter."
      />

      <ComparisonTable
        headers={["", "Parameters", "Hyperparameters"]}
        rows={[
          ["Set by", "The training algorithm", "You, or an automated search"],
          ["When", "During training", "Before training"],
          ["Optimised against", "The training loss", "Validation performance"],
          ["Saved with the model", "Yes — they are the model", "Recorded as config"],
          ["Linear / logistic regression", "Coefficients w, intercept b", "λ / C, penalty type, solver, max_iter"],
          ["Decision tree", "The split feature and threshold at each node, leaf values", "max_depth, min_samples_leaf, criterion, ccp_alpha"],
          ["Random forest", "Every tree's structure", "n_estimators, max_features, max_depth, bootstrap"],
          ["Gradient boosting", "Every tree's structure and leaf weights", "learning_rate, n_estimators, max_depth, subsample, colsample_bytree, reg_lambda"],
          ["Neural network", "Weights and biases", "learning rate, architecture, batch size, dropout rate, optimizer, epochs"],
          ["SVM", "Support vectors and their dual coefficients", "C, kernel, gamma"],
          ["KNN", "None — it stores the training set", "K, distance metric, weighting"],
        ]}
      />

      <SubHeading priority="must">Search strategies</SubHeading>
      <Grid cols={3}>
        <Panel title="Grid Search">
          <p className="mb-2">Try every combination on a predefined grid.</p>
          <ul className="bullet-list">
            <li>Exhaustive and reproducible.</li>
            <li>Cost explodes combinatorially: 4 hyperparameters × 5 values = 625 fits, × 5 CV folds = 3,125 trainings.</li>
            <li>Fine for 1–2 hyperparameters with a small, well-understood range.</li>
          </ul>
        </Panel>
        <Panel title="Random Search">
          <p className="mb-2">Sample combinations at random from distributions you specify.</p>
          <ul className="bullet-list">
            <li>You control the budget directly — run 60 trials, stop.</li>
            <li>Usually beats grid search at equal budget (see below).</li>
            <li>Sample the learning rate and λ log-uniformly, not uniformly.</li>
          </ul>
        </Panel>
        <Panel title="Bayesian Optimization">
          <p className="mb-2">Build a probabilistic model of the objective and pick the next point to try where the expected improvement is highest.</p>
          <ul className="bullet-list">
            <li>Far more sample-efficient when each training run is expensive.</li>
            <li>Optuna, Hyperopt, scikit-optimize.</li>
            <li>Sequential by nature, so it parallelises less cleanly than random search.</li>
            <li>Related: Hyperband / successive halving, which kill bad configurations early.</li>
          </ul>
        </Panel>
      </Grid>

      <DefinitionCard term="Why random search usually beats grid search">
        <p className="mb-2">
          The key observation (Bergstra &amp; Bengio, 2012) is that in most problems only <strong>a few
          hyperparameters actually matter</strong>, and you don't know in advance which ones.
        </p>
        <Formula>{`Budget: 9 trials, 2 hyperparameters,
        only the FIRST one matters.

GRID (3×3)                RANDOM (9 points)
  x  x  x                   x    x   x
  x  x  x                    x  x      x
  x  x  x                  x      x  x

→ 3 distinct values       → 9 distinct values
  of the important one      of the important one`}</Formula>
        <p className="mt-2">
          The grid wastes trials re-testing the same value of the important hyperparameter with different values of
          an irrelevant one. Random search gives you a distinct value of <em>every</em> hyperparameter on every
          trial, so it explores the dimension that matters far more finely at the same cost.
        </p>
      </DefinitionCard>

      <Callout variant="tip" title="What to tune first, and what to leave alone">
        <div className="flex flex-wrap gap-2">
          <Chip tone="red">GBM: learning_rate + n_estimators together</Chip>
          <Chip tone="red">GBM: max_depth / num_leaves</Chip>
          <Chip tone="amber">GBM: subsample, colsample, min_child_weight</Chip>
          <Chip tone="red">Random forest: max_features</Chip>
          <Chip tone="green">Random forest: n_estimators — just set it high</Chip>
          <Chip tone="red">Linear/SVM: C (or λ)</Chip>
          <Chip tone="red">SVM RBF: gamma</Chip>
          <Chip tone="red">Neural net: learning rate, above all else</Chip>
        </div>
        <p className="mt-2">
          A good line to have ready: "I'd tune the learning rate first — it's almost always the highest-leverage
          knob — and for a random forest I wouldn't tune <Code>n_estimators</Code> at all, because more trees never
          hurts accuracy, only runtime."
        </p>
      </Callout>
    </>
  );
}
