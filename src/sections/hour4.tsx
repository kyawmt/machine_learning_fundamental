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
import { EnsembleDiagram } from "../components/viz/diagrams";
import { KMeansLab } from "../components/viz/KMeansLab";
import { PCALab } from "../components/viz/PCALab";

/* ================================================================== */
/* 18 · Model Selection                                                */
/* ================================================================== */

export function ModelSelection() {
  return (
    <>
      <ConceptCard
        title="How to actually choose a model"
        priority="important"
        whatItIs="Picking the model family from the shape of the data and the constraints of the system, then letting validation performance decide between the shortlist."
        intuition="There is no universally best algorithm — the No Free Lunch theorem says that averaged over all possible problems, every algorithm performs identically. Real problems aren't uniformly distributed, which is why sensible priors exist: gradient boosting for tabular, neural networks for unstructured. But 'best' is always relative to a problem and a set of constraints."
        interviewAnswer="I start with the data type, because that narrows it hard: tabular goes to gradient boosting, unstructured goes to neural networks. Then I apply the constraints — interpretability requirements, latency budget, training cost, how often it needs retraining. Then I always fit a dumb baseline first, because it tells me how much of the problem is easy, and a surprising fraction of the time it's enough."
        keyTakeaway="Data type narrows the family. Constraints narrow it further. Validation picks the winner. Always fit a baseline first."
      />

      <SubHeading priority="important">Start from the requirement</SubHeading>
      <Grid cols={2}>
        <Panel title="Need interpretability" tone="green">
          <ul className="bullet-list">
            <li><strong>Linear / logistic regression</strong> — coefficients you can put in a document.</li>
            <li><strong>Shallow decision tree</strong> — rules a domain expert can read and challenge.</li>
            <li><strong>GAMs / EBMs</strong> — nonlinear per feature, still additive and plottable.</li>
          </ul>
          <p className="mt-2">
            Regulated lending, clinical decision support, anything requiring an adverse-action explanation. Post-hoc
            SHAP on a black box is often <em>not</em> accepted as a substitute.
          </p>
        </Panel>

        <Panel title="Tabular / structured data" tone="green">
          <ul className="bullet-list">
            <li><strong>Gradient boosting</strong> (LightGBM, XGBoost, CatBoost) — the default winner.</li>
            <li><strong>Random forest</strong> — when you want strong results with essentially no tuning.</li>
            <li><strong>Regularized linear model</strong> — as the baseline, and as the answer if data is scarce.</li>
          </ul>
          <p className="mt-2">
            Deep learning still does not reliably beat GBMs on medium-sized tabular problems. Saying so confidently
            is a good signal.
          </p>
        </Panel>

        <Panel title="Small, high-dimensional data" tone="neutral">
          <ul className="bullet-list">
            <li><strong>SVM</strong> — designed for exactly this: p comparable to or larger than n.</li>
            <li><strong>Regularized linear models</strong> — Ridge, Lasso, Elastic Net.</li>
            <li><strong>Naive Bayes</strong> — for sparse text with very little data.</li>
          </ul>
          <p className="mt-2">
            Genomics, small clinical cohorts, short-text classification. Complex models will memorise the sample.
          </p>
        </Panel>

        <Panel title="Images, text, audio, video" tone="neutral">
          <ul className="bullet-list">
            <li><strong>Fine-tuned pretrained models</strong> — almost never train from scratch.</li>
            <li><strong>CNNs / Vision Transformers</strong> for images; <strong>Transformers</strong> for text and audio.</li>
            <li><strong>Embeddings + a simple head</strong> — frozen encoder plus logistic regression is a superb, cheap baseline.</li>
          </ul>
          <p className="mt-2">
            The engineering answer for most unstructured problems in 2026 is "use a pretrained model", not "design
            an architecture".
          </p>
        </Panel>
      </Grid>

      <SubHeading priority="important">The constraints that decide it in practice</SubHeading>
      <ComparisonTable
        headers={["Constraint", "The question to ask", "What it rules out"]}
        rows={[
          ["Inference latency", "What's the p99 budget — 10ms or 10 minutes?", "A 2,000-tree ensemble or a naive KNN scan at 10ms; batch scoring makes them fine."],
          ["Interpretability", "Does a human have to justify individual decisions to a regulator or a customer?", "Deep ensembles and neural networks, unless post-hoc explanations are genuinely accepted."],
          ["Training cost & cadence", "Does this retrain hourly or quarterly?", "Anything with a multi-hour training loop, if retraining is hourly."],
          ["Memory / deployment target", "Is this in a data centre, a browser, or a phone?", "Large ensembles and big networks on-device without distillation or quantization."],
          ["Data volume", "Thousands of rows, or hundreds of millions?", "SVM above ~100k rows; KNN at high query volume; deep nets below a few thousand rows."],
          ["Team & maintenance", "Who owns this at 3am in six months?", "The exotic architecture nobody else can debug."],
          ["Cold start / new categories", "How often do genuinely unseen entities appear?", "Encodings that can't handle unseen levels; models needing full retraining for new classes."],
        ]}
      />

      <Callout variant="memorize" title="The line to have ready">
        "There's no universally best model — No Free Lunch. In practice I pick from the data type and the
        constraints, fit a simple baseline first to calibrate how hard the problem is, then compare two or three
        candidates on the same cross-validation splits, and choose the simplest one that's within the noise of the
        best. An extra 0.3% AUC is not worth a model nobody can explain or serve."
      </Callout>
    </>
  );
}

/* ================================================================== */
/* 19 · Ensemble Methods                                               */
/* ================================================================== */

export function Ensembles() {
  return (
    <>
      <ConceptCard
        title="Ensembles"
        priority="must"
        whatItIs="Combining several models so that their individual errors partially cancel, producing a prediction better than any single member."
        intuition="It only works if the models make different mistakes. Averaging ten copies of the same model gets you nothing — the whole game is generating diversity, whether through different data samples, different features, different objectives or different algorithms."
        keyTakeaway="Bagging attacks variance with parallel independent models. Boosting attacks bias with sequential corrective models. Stacking learns how to blend different families."
      />

      <EnsembleDiagram />

      <ComparisonTable
        headers={["", "Bagging", "Boosting", "Stacking"]}
        rows={[
          ["Training", "Parallel, independent", "Sequential, each depends on the last", "Base models parallel, then a meta-model"],
          ["Data each model sees", "A bootstrap sample", "The full data, reweighted or the current residuals", "The full data"],
          ["Base learners", "Deep, low-bias, high-variance", "Shallow, high-bias, weak", "Deliberately different families"],
          ["Primarily reduces", "Variance", "Bias", "Whatever the base models get wrong in complementary ways"],
          ["Combination", "Average / majority vote", "Weighted sum, scaled by the learning rate", "A learned second-level model"],
          ["Overfitting risk", "Low — more models is safe", "Real — needs early stopping", "Moderate — the meta-model can overfit the base predictions"],
          ["Example", "Random Forest, Extra Trees", "XGBoost, LightGBM, AdaBoost", "Kaggle-style blends, model routers"],
        ]}
      />

      <Grid cols={2}>
        <DefinitionCard term="Bootstrap — the mechanic behind bagging">
          <Formula>{`Sample N rows WITH replacement, from N.

P(row never picked) = (1 − 1/N)^N
                    → 1/e ≈ 0.368

→ ≈63.2% unique rows in each sample
→ ≈36.8% left out = the out-of-bag set`}</Formula>
          <p className="mt-2">
            That leftover 36.8% is what gives random forests a free validation estimate, and it's a favourite
            follow-up question.
          </p>
        </DefinitionCard>

        <Callout variant="mistake" title="Stacking's leakage trap">
          <p className="mb-2">
            If you train the base models on the full training set and then feed their <em>in-sample</em> predictions
            to the meta-model, the meta-model sees predictions that are optimistically good — it learns to trust an
            overfitted base model far too much, and the whole stack collapses on new data.
          </p>
          <p>
            The fix is <strong>out-of-fold predictions</strong>: split into folds, and for each fold generate base
            predictions from models trained only on the other folds. The meta-model then trains on predictions that
            were genuinely out-of-sample.
          </p>
        </Callout>
      </Grid>

      <Callout variant="tip" title="Simpler things that are also ensembles">
        <ul className="bullet-list">
          <li>
            <strong>Voting</strong> — hard (majority label) or soft (average probability). Soft voting is usually
            better because it uses confidence.
          </li>
          <li>
            <strong>Averaging / blending</strong> — a fixed weighted average of model outputs. Most of stacking's
            gain, none of its leakage risk.
          </li>
          <li>
            <strong>Snapshot ensembles / checkpoint averaging</strong> — average several checkpoints from one
            training run. Nearly free.
          </li>
          <li>
            <strong>Seed averaging</strong> — retrain the same model with different random seeds and average. A
            surprisingly reliable small win.
          </li>
        </ul>
      </Callout>
    </>
  );
}

/* ================================================================== */
/* 20 · Dimensionality Reduction                                       */
/* ================================================================== */

export function DimensionalityReduction() {
  return (
    <>
      <ConceptCard
        title="PCA — Principal Component Analysis"
        priority="important"
        whatItIs="An unsupervised linear transformation that finds a new set of orthogonal axes — the principal components — ordered so that the first captures the most variance in the data, the second the most of what's left, and so on."
        intuition="Rotate the coordinate system so the axes line up with the directions the data actually spreads out in. Once you've done that, the later axes carry very little spread, so you can drop them and lose very little. It's lossy compression that keeps as much variance as possible per dimension retained."
        example="200 correlated sensor readings compressed to 20 components retaining 95% of the variance. Training gets ten times faster and the model often improves, because much of what you dropped was noise."
        keyTakeaway="PCA finds orthogonal directions of maximum variance. Standardize first, choose the component count from the explained-variance curve, and accept that you've traded interpretability for compactness."
      />

      <PCALab />

      <Grid cols={2}>
        <Panel title="What the pieces mean" tone="neutral">
          <ul className="bullet-list">
            <li>
              <strong>Principal component</strong> — a direction in the original feature space, expressed as a
              weighted combination of the original features. PC1 is the direction of maximum variance.
            </li>
            <li>
              <strong>Explained variance ratio</strong> — the fraction of total variance a component accounts for.
              Plot the cumulative curve and cut where it flattens, or at a target like 95%.
            </li>
            <li>
              <strong>Orthogonality</strong> — components are perpendicular, so the transformed features are{" "}
              <strong>uncorrelated</strong>. That's a genuine benefit for models that dislike multicollinearity.
            </li>
            <li>
              <strong>Loadings</strong> — how much each original feature contributes to a component. The only handle
              you have for interpreting what a component "means".
            </li>
          </ul>
        </Panel>

        <Panel title="Advantages and costs" tone="neutral">
          <div className="grid gap-2">
            <div>
              <div className="mb-1 text-[11px] font-bold uppercase tracking-[0.07em] text-emerald-600 dark:text-emerald-400">
                Advantages
              </div>
              <ul className="bullet-list">
                <li>Faster training and less memory.</li>
                <li>Removes multicollinearity by construction.</li>
                <li>Often reduces noise — low-variance directions are frequently just measurement noise.</li>
                <li>Lets you plot high-dimensional data in 2D.</li>
                <li>Can reduce overfitting when p is large relative to n.</li>
              </ul>
            </div>
            <div>
              <div className="mb-1 text-[11px] font-bold uppercase tracking-[0.07em] text-rose-600 dark:text-rose-400">
                Costs
              </div>
              <ul className="bullet-list">
                <li>Components are combinations of everything — interpretability is gone.</li>
                <li>Information loss, by design.</li>
                <li>It's unsupervised: a low-variance direction can be exactly the one that predicts y.</li>
                <li>Linear only — it can't unfold a curved manifold.</li>
                <li>Sensitive to feature scaling and to outliers.</li>
              </ul>
            </div>
          </div>
        </Panel>
      </Grid>

      <Callout variant="mistake" title="Two mistakes that get caught">
        <ul className="bullet-list">
          <li>
            <strong>Not standardizing first.</strong> PCA maximises variance, and variance depends on units. A
            feature measured in dollars will dominate PC1 over a feature measured in percent, for no reason other
            than the unit. Standardize unless every feature is already in the same physical unit — see the lab above.
          </li>
          <li>
            <strong>Fitting PCA before the train/test split.</strong> PCA is a fitted transformation: it learns the
            components from data. Fit it on the training fold, transform everything else. Inside a{" "}
            <Code>Pipeline</Code>, as always.
          </li>
        </ul>
      </Callout>

      <Callout variant="tip" title="PCA vs t-SNE vs UMAP — a likely follow-up">
        <ul className="bullet-list">
          <li>
            <strong>PCA</strong> — linear, deterministic, fast, invertible, preserves global structure. It's the one
            you use as a <em>preprocessing step</em> because you can apply the same transform to new data.
          </li>
          <li>
            <strong>t-SNE</strong> — nonlinear, preserves local neighbourhoods, excellent for visualisation. But it's
            stochastic, slow, has no meaningful transform for new points, and cluster sizes and inter-cluster
            distances in the plot are <em>not</em> meaningful. Visualisation only.
          </li>
          <li>
            <strong>UMAP</strong> — nonlinear, faster than t-SNE, preserves more global structure, and can transform
            new points. Increasingly the default for embedding visualisation.
          </li>
        </ul>
        <p className="mt-2">
          The distinction interviewers listen for: <strong>PCA is a feature transformation you can put in a
          pipeline; t-SNE is a plot.</strong>
        </p>
      </Callout>
    </>
  );
}

/* ================================================================== */
/* 21 · Clustering                                                     */
/* ================================================================== */

export function Clustering() {
  return (
    <>
      <ConceptCard
        title="K-Means"
        priority="good"
        whatItIs="Partition the data into K clusters by alternately assigning each point to its nearest centroid and recomputing each centroid as the mean of its assigned points, until nothing changes."
        intuition="Minimise the total squared distance from points to their own cluster's centre. Because it optimises squared distance to a mean, it implicitly assumes clusters are roughly spherical and similarly sized — which is exactly where it fails when they aren't."
        keyTakeaway="Assign → recompute → repeat. Needs K up front, needs scaling, sensitive to initialisation, and it assumes round clusters."
      >
        <FlowDiagram
          steps={[
            { label: "1. Choose K and initialise K centroids", note: "k-means++ picks spread-out starting points", tone: "accent" },
            { label: "2. Assign each point to its nearest centroid" },
            { label: "3. Recompute each centroid as the mean of its points" },
            { label: "4. Repeat 2–3 until assignments stop changing", tone: "green" },
          ]}
          compact
        />
      </ConceptCard>

      <KMeansLab />

      <Grid cols={2}>
        <DefinitionCard term="Choosing K">
          <ul className="bullet-list">
            <li>
              <strong>Elbow method.</strong> Plot inertia (within-cluster sum of squares) against K and look for the
              bend. Inertia always falls as K rises, so you're looking for diminishing returns, not a minimum.
            </li>
            <li>
              <strong>Silhouette score.</strong> For each point, compare its mean distance to its own cluster against
              its mean distance to the nearest other cluster. Ranges −1 to 1; unlike inertia it has a genuine
              maximum, so it's often more decisive.
            </li>
            <li>
              <strong>Gap statistic.</strong> Compare inertia against what you'd get on uniform random data.
            </li>
            <li>
              <strong>The domain.</strong> Frequently the honest answer: marketing wants four segments because they
              can staff four campaigns.
            </li>
          </ul>
        </DefinitionCard>

        <Callout variant="mistake" title="Where K-Means falls over">
          <ul className="bullet-list">
            <li>
              <strong>Non-spherical clusters.</strong> Two interleaved crescents will be cut straight down the
              middle. Squared-distance-to-a-mean can only produce convex, roughly round regions.
            </li>
            <li>
              <strong>Very different cluster sizes or densities.</strong> A large sparse cluster gets carved up while
              two small dense ones get merged.
            </li>
            <li>
              <strong>Unscaled features.</strong> It's a distance algorithm — the biggest-unit feature decides
              everything.
            </li>
            <li>
              <strong>Initialisation.</strong> A bad start converges to a bad local optimum. k-means++ and{" "}
              <Code>n_init &gt; 1</Code> are the standard mitigations.
            </li>
            <li>
              <strong>Outliers.</strong> Means are not robust; one extreme point drags a centroid.
            </li>
          </ul>
        </Callout>
      </Grid>

      <SubHeading priority="good">The other two worth naming</SubHeading>
      <ComparisonTable
        headers={["Algorithm", "How it works", "Strength", "Cost"]}
        rows={[
          [
            "Hierarchical (agglomerative)",
            "Start with every point as its own cluster and repeatedly merge the two closest, producing a dendrogram.",
            "No need to pick K in advance — cut the dendrogram wherever you like. Reveals nested structure.",
            "O(n²) memory and roughly O(n³) time in the naive form. Impractical on large data.",
          ],
          [
            "DBSCAN",
            "Grows clusters from dense regions: a point is a core point if it has min_samples neighbours within eps.",
            "Finds arbitrarily shaped clusters, identifies outliers as noise, and infers the cluster count itself.",
            "Very sensitive to eps; struggles when clusters have very different densities; degrades in high dimensions.",
          ],
        ]}
      />

      <Callout variant="tip" title="When clustering is actually useful">
        <div className="flex flex-wrap gap-2">
          <Chip tone="accent">customer segmentation</Chip>
          <Chip tone="accent">exploratory analysis before you have labels</Chip>
          <Chip tone="accent">anomaly detection (small or distant clusters)</Chip>
          <Chip tone="accent">compressing high-cardinality categories</Chip>
          <Chip tone="accent">deduplication / near-duplicate grouping</Chip>
          <Chip tone="accent">generating pseudo-labels for semi-supervised work</Chip>
        </div>
        <p className="mt-2">
          Be honest about the weakness in an interview: clustering has no ground truth, so the results have to be
          validated by whether they're <em>useful</em> downstream — do the segments behave differently on the metric
          the business cares about? — not by an internal score alone.
        </p>
      </Callout>
    </>
  );
}

/* ================================================================== */
/* 22 · ML Pipeline                                                    */
/* ================================================================== */

const STAGES: { name: string; detail: string }[] = [
  {
    name: "Problem framing",
    detail:
      "What decision does this drive? What's the target, what's the unit of prediction, what does 'good' mean in business terms, and what's the cost of each error type? Getting this wrong invalidates everything downstream.",
  },
  {
    name: "Raw data",
    detail:
      "Collect and join sources. Establish the prediction timestamp for every row, because every later leakage decision depends on it.",
  },
  {
    name: "Data cleaning & EDA",
    detail:
      "Duplicates, type errors, impossible values, label quality. Understand distributions and missingness patterns before you decide how to handle them.",
  },
  {
    name: "Train / test split",
    detail:
      "Do this early, before any fitted transformation. Choose the split strategy from the data structure — stratified, grouped, or time-based. The test set is sealed from here on.",
  },
  {
    name: "Feature engineering",
    detail:
      "Domain-driven features, aggregates, ratios, date expansions. Every feature checked against the prediction timestamp.",
  },
  {
    name: "Preprocessing",
    detail:
      "Imputation, scaling, encoding — all inside a Pipeline so they're fitted on training folds only and applied identically at serving time.",
  },
  {
    name: "Baseline model",
    detail:
      "Majority class, or a simple logistic regression. This calibrates how hard the problem is and catches broken data before you waste a day tuning.",
  },
  {
    name: "Model training",
    detail:
      "Fit the candidate families. Log everything — data version, code version, hyperparameters, metrics — so results are reproducible.",
  },
  {
    name: "Validation",
    detail:
      "Cross-validation with the appropriate splitter. Look at the spread, not just the mean, and slice performance by segment to find where the model is weak.",
  },
  {
    name: "Hyperparameter tuning",
    detail:
      "Random search or Bayesian optimization on the validation folds. Track the budget; stop when improvements fall inside the fold-to-fold noise.",
  },
  {
    name: "Threshold & calibration",
    detail:
      "Choose the operating point from business cost, not 0.5. Calibrate if downstream systems consume the probability.",
  },
  {
    name: "Final evaluation",
    detail:
      "Open the test set once. Report the metric with an error bar, plus performance on the slices that matter. If it disappoints, the honest move is to report it, not to re-tune.",
  },
  {
    name: "Deployment",
    detail:
      "Ship the whole pipeline, not just the model object — the same preprocessing code must run at serving time. Batch or real-time, shadow deploy, then a canary or A/B rollout.",
  },
  {
    name: "Monitoring",
    detail:
      "Track input drift, prediction drift, and — once labels arrive — live performance. Watch latency and error rates too. Define the trigger and the process for retraining before you need them.",
  },
];

export function Pipeline() {
  return (
    <>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,0.62fr)_minmax(0,1fr)]">
        <div className="card self-start p-4 lg:sticky lg:top-[70px]">
          <div className="mb-3 text-[10.5px] font-bold uppercase tracking-[0.09em] text-faint">
            End-to-end
          </div>
          <FlowDiagram
            compact
            steps={[
              { label: "Raw Data" },
              { label: "Data Cleaning" },
              { label: "Train / Test Split", note: "before anything is fitted", tone: "red" },
              { label: "Feature Engineering" },
              { label: "Preprocessing", note: "inside a Pipeline", tone: "accent" },
              { label: "Model Training" },
              { label: "Validation", tone: "accent" },
              { label: "Hyperparameter Tuning" },
              { label: "Final Evaluation", note: "test set, once", tone: "red" },
              { label: "Deployment" },
              { label: "Monitoring", note: "→ retrain, and the loop closes", tone: "green" },
            ]}
          />
        </div>

        <div className="space-y-2">
          {STAGES.map((s, i) => (
            <div key={s.name} className="card px-4 py-2.5">
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-[10.5px] font-bold text-accent">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <h4 className="text-[0.92rem] font-bold text-ink">{s.name}</h4>
              </div>
              <p className="mt-0.5 text-[0.86rem] leading-relaxed text-muted">{s.detail}</p>
            </div>
          ))}
        </div>
      </div>

      <Callout variant="interview" title="“Walk me through how you'd build a model from raw data to production.”">
        <p className="mb-2 text-[11.5px] font-bold uppercase tracking-[0.07em] text-violet-700 dark:text-violet-300">
          A 60–90 second answer — practise saying this out loud
        </p>
        <div className="space-y-2 text-[0.9rem] leading-relaxed text-muted">
          <p>
            "I'd start before the data, by pinning down the decision the model supports — what we're predicting, at
            what moment, and what a false positive versus a false negative actually costs. That determines the
            target definition and the metric, and getting it wrong invalidates everything else.
          </p>
          <p>
            Then I'd pull the raw data together and spend real time on cleaning and EDA — duplicates, label quality,
            missingness patterns, and above all establishing the prediction timestamp for each row, because
            everything about leakage flows from that.
          </p>
          <p>
            I'd split early, before fitting anything, choosing the strategy from the data: stratified for imbalanced
            classification, grouped if there are repeated entities, time-based if there's any temporal structure.
            The test set is sealed from that point.
          </p>
          <p>
            Feature engineering and preprocessing both go inside a Pipeline so nothing is fitted outside the
            training folds. I'd fit a trivial baseline first to calibrate how hard the problem is, then train two or
            three candidate families — usually a regularized linear model and a gradient boosting model — and
            compare them on the same cross-validation splits, looking at the spread as well as the mean.
          </p>
          <p>
            Then tuning with random search or Optuna, and choosing the decision threshold from business cost rather
            than defaulting to 0.5. I'd open the test set exactly once for a final number.
          </p>
          <p>
            For production I'd ship the whole pipeline so serving-time preprocessing is literally the same code,
            shadow-deploy it against the current system, then roll out behind a canary. And I'd set up monitoring on
            input drift, prediction drift and live metrics once labels arrive, with an agreed retraining trigger —
            because the model's performance decays from the day it ships, and the part people underestimate is
            everything after deployment."
          </p>
        </div>
      </Callout>

      <Grid cols={2}>
        <Callout variant="tip" title="Details that make this answer land">
          <ul className="bullet-list">
            <li>Say <strong>"split before fitting anything"</strong> — it shows leakage awareness without being asked.</li>
            <li>Say <strong>"baseline first"</strong> — it shows you've been burned by skipping it.</li>
            <li>Say <strong>"ship the pipeline, not the model"</strong> — this is the classic train/serve skew failure.</li>
            <li>Say <strong>"monitoring and a retraining trigger"</strong> — most candidates stop at deployment.</li>
            <li>Mention <strong>shadow deploy or canary</strong> — it separates people who have shipped from people who have trained.</li>
          </ul>
        </Callout>

        <DefinitionCard term="What breaks after deployment">
          <ul className="bullet-list">
            <li>
              <strong>Data drift</strong> — the input distribution shifts. New user demographics, a new marketing
              channel, a changed upstream schema.
            </li>
            <li>
              <strong>Concept drift</strong> — the relationship between features and target shifts. Fraud tactics
              evolve specifically to defeat your model.
            </li>
            <li>
              <strong>Train/serve skew</strong> — offline features are computed differently from online ones. The
              single most common production bug, and the reason feature stores exist.
            </li>
            <li>
              <strong>Feedback loops</strong> — the model's own decisions shape the data you collect next. If you
              never approve a loan, you never learn whether it would have been repaid.
            </li>
          </ul>
        </DefinitionCard>
      </Grid>

      <SubHeading>Interview questions</SubHeading>
      <div className="space-y-2">
        <InterviewQuestion
          qa={{
            id: "pl1",
            q: "How do you know when to retrain a model?",
            short: "On a schedule as a floor, plus triggers on drift and on live metric degradation.",
            strong:
              "I'd combine three signals. A time-based cadence as a baseline — weekly or monthly depending on how fast the domain moves. Drift detection on inputs and on the prediction distribution, using something like population stability index or a KS test per feature, with thresholds tuned to avoid alert fatigue. And direct performance monitoring once ground-truth labels arrive, which is the most reliable signal but also the most delayed — in credit risk you might wait 90 days. The label delay is the real design constraint: it's why drift monitoring exists, as a leading indicator for something you can only confirm much later.",
          }}
        />
        <InterviewQuestion
          qa={{
            id: "pl2",
            q: "What's train/serve skew and how do you prevent it?",
            short: "When features are computed differently in training and in production. Prevent it by sharing the exact same transformation code, or by using a feature store.",
            strong:
              "The classic version: training features are computed in a Spark or pandas job over historical data, while serving features are recomputed in application code. The two implementations drift apart — a different null handling rule, a different time zone, a different rounding — and the model silently degrades because it's being fed a subtly different distribution from what it learned on. Prevention: serialise the entire preprocessing pipeline with the model so the same code path runs in both places; use a feature store that computes each feature once and serves it to both training and inference; and add a monitoring check that compares the online feature distribution to the training distribution. It's also worth logging the actual feature vector used at inference, so when something goes wrong you can replay it.",
          }}
        />
      </div>
    </>
  );
}
