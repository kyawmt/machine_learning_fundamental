import { Callout } from "../components/Callout";
import { DefinitionCard } from "../components/DefinitionCard";
import { ConceptCard } from "../components/ConceptCard";
import { InterviewQuestion } from "../components/InterviewQuestion";
import {
  Chip,
  Code,
  ComparisonTable,
  Formula,
  Grid,
  Panel,
  SubHeading,
  YesNo,
} from "../components/ui";

/* ================================================================== */
/* 09 · Feature Engineering                                            */
/* ================================================================== */

export function FeatureEngineering() {
  return (
    <>
      <ConceptCard
        title="Feature Engineering"
        priority="important"
        whatItIs="Transforming raw columns into representations that make the pattern easier for the model to find — usually by injecting domain knowledge the algorithm has no way to discover on its own."
        intuition="A model can only combine what you give it. If the real driver is 'spend per visit' and you supply total spend and visit count separately, a linear model will never find the ratio — but a tree will need many splits to approximate it. Handing it the ratio directly is worth more than most hyperparameter tuning."
        interviewAnswer="On tabular problems, feature engineering usually beats model selection. I start from the domain: what would a human expert look at to make this call? Then I build those quantities explicitly — ratios, rates, time-since-last-event, aggregates over the right window — and check each one for leakage against the prediction timestamp."
        keyTakeaway="Good features > fancy models on tabular data. And every engineered feature is a leakage risk until you've checked its timing."
      />

      <SubHeading priority="must">The standard moves</SubHeading>
      <Grid cols={2}>
        <Panel title="Dates — the highest-yield expansion">
          <Formula>{`raw:  order_ts = 2026-08-22T19:14:00

→ hour            = 19
→ day_of_week     = 5 (Saturday)
→ is_weekend      = 1
→ day_of_month    = 22
→ month           = 8
→ quarter         = 3
→ is_month_end    = 0
→ is_holiday      = 0
→ days_since_signup = 412
→ days_since_last_order = 3`}</Formula>
          <p className="mt-2">
            The last two are usually the most predictive of all, and they're the ones people forget. For cyclical
            fields, encode as <Code>sin(2π·h/24)</Code> and <Code>cos(2π·h/24)</Code> so hour 23 sits next to hour 0.
          </p>
        </Panel>

        <Panel title="Numeric transformations">
          <ul className="bullet-list">
            <li>
              <strong>Log / log1p</strong> — for right-skewed, positive quantities (income, prices, counts). Compresses
              the long tail so it stops dominating a distance- or variance-based model.
            </li>
            <li>
              <strong>Ratios and rates</strong> — <Code>spend / visits</Code>, <Code>errors / requests</Code>,{" "}
              <Code>debt / income</Code>. Almost always more informative than the raw parts.
            </li>
            <li>
              <strong>Differences</strong> — <Code>current − rolling_mean_30d</Code> turns a level into a deviation.
            </li>
            <li>
              <strong>Binning</strong> — turn age into brackets to let a linear model express a non-monotone effect.
              Costs resolution; rarely needed for trees.
            </li>
            <li>
              <strong>Polynomial / interaction terms</strong> — <Code>x₁x₂</Code>, <Code>x²</Code>. Gives linear
              models curvature, at the cost of a fast-growing feature count.
            </li>
            <li>
              <strong>Clipping / winsorizing</strong> — cap at the 1st and 99th percentile so a single absurd value
              doesn't dominate.
            </li>
          </ul>
        </Panel>

        <Panel title="Aggregations — where the real signal usually lives">
          <ul className="bullet-list">
            <li>Per-entity counts, means, maxima, standard deviations over a window.</li>
            <li>
              <Code>transactions_last_7d</Code>, <Code>avg_basket_90d</Code>,{" "}
              <Code>distinct_merchants_24h</Code>.
            </li>
            <li>Deviation from the entity's own history: <Code>amount / user_avg_amount</Code>.</li>
            <li>Deviation from a peer group: <Code>price / median_price_in_category</Code>.</li>
          </ul>
          <p className="mt-2 text-rose-600 dark:text-rose-400">
            Every one of these is a leakage risk. The window must end strictly before the prediction timestamp.
          </p>
        </Panel>

        <Panel title="Text and categorical shortcuts">
          <ul className="bullet-list">
            <li>Length, word count, digit ratio, uppercase ratio — cheap and often surprisingly predictive.</li>
            <li>TF-IDF for classic models; sentence embeddings when semantic similarity matters.</li>
            <li>Count / frequency encoding: replace a category with how often it appears.</li>
            <li>Missing-value indicators (see the next section) are engineered features too.</li>
          </ul>
        </Panel>
      </Grid>

      <Callout variant="mistake" title="The three ways feature engineering leaks">
        <ul className="bullet-list">
          <li>
            <strong>Windows that straddle the prediction point.</strong> <Code>avg_spend_last_30d</Code> computed from
            a static snapshot rather than as-of the event time.
          </li>
          <li>
            <strong>Statistics fitted on all rows.</strong> Category means, global medians, TF-IDF vocabularies, PCA
            components — all of these must be learned inside the training fold.
          </li>
          <li>
            <strong>Features that only exist because the outcome happened.</strong>{" "}
            <Code>refund_amount</Code> when predicting fraud; <Code>cancellation_reason</Code> when predicting churn.
          </li>
        </ul>
      </Callout>

      <Callout variant="tip">
        Interviewers love the follow-up "how do you know a feature is worth keeping?" Good answer: permutation
        importance or SHAP on a validation set, plus a plain ablation — retrain without it and see whether the
        validation metric actually moves beyond the CV standard deviation. Correlation with the target alone is not
        enough, because it misses interactions and rewards leakage.
      </Callout>
    </>
  );
}

/* ================================================================== */
/* 10 · Feature Scaling                                                */
/* ================================================================== */

export function FeatureScaling() {
  return (
    <>
      <ConceptCard
        title="Feature Scaling"
        priority="must"
        whatItIs="Putting numeric features onto comparable ranges so that no feature dominates purely because of the unit it happens to be measured in."
        intuition="Income in dollars runs to 10⁵; age runs to 10². Any algorithm that computes distances or sums weighted inputs will treat income as a hundred times more important than age — not because it is, but because dollars are small units."
        example="KNN on {age: 35, income: 85000} vs {age: 60, income: 85200}. Euclidean distance is 200.0 — the 25-year age difference contributes almost nothing. After standardizing, age becomes the dominant difference, which is what you actually wanted."
        keyTakeaway="Scaling matters for distance-based, gradient-based and variance-based methods. It's irrelevant to tree splits."
      />

      <Grid cols={2}>
        <Panel title="Standardization (z-score)">
          <Formula>{`z = (x − μ) / σ`}</Formula>
          <ul className="bullet-list mt-2">
            <li>Result has mean 0 and standard deviation 1; the range is unbounded.</li>
            <li>Preserves the shape of the distribution, including outliers.</li>
            <li>
              <strong>The default choice.</strong> Required by PCA, and what linear models and neural networks
              generally expect.
            </li>
            <li><Code>StandardScaler</Code></li>
          </ul>
        </Panel>
        <Panel title="Min-Max Normalization">
          <Formula>{`x' = (x − min) / (max − min)`}</Formula>
          <ul className="bullet-list mt-2">
            <li>Squashes everything into [0, 1].</li>
            <li>
              <strong>Very sensitive to outliers</strong> — one extreme value compresses everything else into a
              sliver of the range.
            </li>
            <li>Use when you need a bounded range: image pixels, some neural-net inputs, certain distance measures.</li>
            <li><Code>MinMaxScaler</Code></li>
          </ul>
        </Panel>
        <Panel title="Robust scaling">
          <Formula>{`x' = (x − median) / IQR`}</Formula>
          <ul className="bullet-list mt-2">
            <li>Centres on the median and scales by the interquartile range.</li>
            <li>The right pick when the feature has genuine outliers you want to keep but not be dominated by.</li>
            <li><Code>RobustScaler</Code></li>
          </ul>
        </Panel>
        <Panel title="Log transform">
          <Formula>{`x' = log(1 + x)`}</Formula>
          <ul className="bullet-list mt-2">
            <li>Not scaling in the same sense — it changes the shape, compressing a right-skewed tail.</li>
            <li>Often the right first step for money, counts and durations, followed by standardization.</li>
            <li>Requires x &gt; −1; use <Code>log1p</Code> so zeros are safe.</li>
          </ul>
        </Panel>
      </Grid>

      <SubHeading priority="must">Which algorithms care</SubHeading>
      <ComparisonTable
        headers={["Algorithm", "Scaling needed?", "Why"]}
        rows={[
          ["KNN", <YesNo key="a" v="yes" />, "Predictions are literally distance computations. Unscaled features silently pick the winner."],
          ["SVM (any kernel)", <YesNo key="b" v="yes" />, "The margin and the RBF kernel both depend on distance. Unscaled SVMs train slowly and fit badly."],
          ["K-Means", <YesNo key="c" v="yes" />, "Centroid assignment is a distance computation."],
          ["PCA", <YesNo key="d" v="yes" />, "Components are directions of maximum variance, and variance is unit-dependent. Skipping this makes PC1 whichever feature has the biggest numbers."],
          ["Neural networks", <YesNo key="e" v="yes" />, "Unscaled inputs give an ill-conditioned loss surface, saturated activations and unstable gradients."],
          ["Linear / logistic regression + L1 or L2", <YesNo key="f" v="yes" />, "The penalty applies equally to every coefficient, so a feature in small units gets an unfairly large penalty."],
          ["Plain linear regression (no penalty)", <YesNo key="g" v="no" />, "Coefficients simply rescale; predictions are identical. Scaling only helps interpretation and numerical conditioning."],
          ["Decision Tree", <YesNo key="h" v="no" />, "Splits are thresholds on one feature at a time. Any monotone rescaling gives the same split."],
          ["Random Forest", <YesNo key="i" v="no" />, "Same reasoning — it's a collection of trees."],
          ["Gradient Boosting (XGBoost, LightGBM, CatBoost)", <YesNo key="j" v="no" />, "Same reasoning. This is a big part of why GBMs are so convenient on messy tabular data."],
          ["Naive Bayes", <YesNo key="k" v="no" />, "Fits a distribution per feature independently; scale is absorbed into the fitted parameters."],
        ]}
      />

      <Callout variant="memorize" title="Why trees genuinely don't care">
        <p className="mb-2">
          A tree asks <Code>is income &lt; 50000?</Code>. Standardize the column and it asks{" "}
          <Code>is income_z &lt; 0.38?</Code> — the <strong>same rows go left</strong>. Splitting only depends on the{" "}
          <em>order</em> of values, and any monotone transformation preserves order. So the tree structure, the
          impurity gains and the predictions are all unchanged.
        </p>
        <p>
          Corollary worth having ready: a log transform doesn't help a tree either, for exactly the same reason. It
          can still help a linear model a great deal.
        </p>
      </Callout>

      <Callout variant="mistake">
        Fitting the scaler on the full dataset before splitting. <Code>fit_transform()</Code> on train,{" "}
        <Code>transform()</Code> on validation and test — and in production, the frozen training statistics get
        applied to a single incoming row. Wrap it in a <Code>Pipeline</Code> and the mistake becomes impossible.
      </Callout>
    </>
  );
}

/* ================================================================== */
/* 11 · Missing Data                                                   */
/* ================================================================== */

export function MissingData() {
  return (
    <>
      <ConceptCard
        title="Handling Missing Data"
        priority="important"
        whatItIs="Deciding what to do about absent values — drop them, fill them, model them, or flag them — with the choice driven by why they're missing."
        intuition="Missing is not the same as zero, and it's not always noise. A blank income field on a loan application might be the single most predictive thing on the form."
        interviewAnswer="First I'd look at how much is missing and whether it's concentrated in particular rows, columns or segments. Then I'd ask why — is it random, or does it correlate with something? For most tabular work I'd use median imputation for numerics and a dedicated 'Missing' category for categoricals, and I'd add a binary missing-indicator whenever the missingness itself looks informative. Everything gets fitted inside the pipeline on training data only."
        keyTakeaway="Ask why it's missing before you decide how to fill it. Add an indicator column when the absence carries information."
      />

      <SubHeading priority="important">Why it's missing — the three regimes</SubHeading>
      <ComparisonTable
        headers={["Mechanism", "Meaning", "Example", "Consequence"]}
        rows={[
          [
            "MCAR",
            "Missing completely at random — unrelated to anything.",
            "A sensor dropped packets at random times.",
            "Dropping rows is unbiased (just wasteful). Simple imputation is safe.",
          ],
          [
            "MAR",
            "Missing at random — explainable by other observed columns.",
            "Income is missing more often for younger applicants.",
            "Imputation conditioned on the other features works well. Dropping rows biases the sample.",
          ],
          [
            "MNAR",
            "Missing not at random — the missingness depends on the unobserved value itself.",
            "High earners decline to state their income.",
            "Any imputation distorts the distribution. The indicator column is essential here.",
          ],
        ]}
        caption="You can rarely prove which regime you're in. But asking the question out loud is exactly what an interviewer wants to hear."
      />

      <SubHeading priority="important">The toolbox</SubHeading>
      <ComparisonTable
        headers={["Approach", "How", "Use when", "Cost"]}
        rows={[
          ["Drop rows", "Delete any row with a missing value.", "Missingness is tiny (<1–2%) and plausibly MCAR.", "Loses data; introduces bias if not MCAR; can't be done at inference time."],
          ["Drop the column", "Remove the feature entirely.", "Most of the column is missing and it carries little signal.", "You may be throwing away the informative pattern of who is missing."],
          ["Mean imputation", "Fill with the training mean.", "Roughly symmetric numeric distribution.", "Shrinks variance; badly distorted by outliers."],
          ["Median imputation", "Fill with the training median.", "The sensible numeric default — skewed data, outliers.", "Still shrinks variance and weakens correlations."],
          ["Mode / 'Missing' category", "Fill with the most common level, or better, an explicit 'Missing' level.", "Categorical features. The explicit level is almost always better.", "Mode imputation invents data; a 'Missing' level does not."],
          ["Constant sentinel", "Fill with −1, 0 or 'Unknown'.", "Tree models, which can split on the sentinel and effectively learn 'was missing'.", "Nonsense for linear models and distance-based ones."],
          ["Model-based (KNN, MICE, IterativeImputer)", "Predict each missing value from the other features.", "Missingness is substantial, features are correlated, accuracy matters.", "Slow; must be fitted inside the CV fold; can leak if done carelessly."],
          ["Native handling", "Let XGBoost / LightGBM / CatBoost learn a default direction for missing at each split.", "You're using a GBM. Frequently the best option and requires nothing.", "Ties you to those libraries."],
          ["Missing indicator", "Add a binary is_missing_x column alongside whatever you filled with.", "Whenever missingness might be informative — which is often.", "One extra column per feature. Cheap insurance."],
        ]}
      />

      <Callout variant="example" title="When the absence is the signal">
        <p className="mb-2">
          On a credit application, <Code>income</Code> is blank for 12% of applicants. You median-impute and move
          on. But blank income correlates with self-employment and with applications abandoned halfway — and both
          predict default. Median imputation erases that entirely; the model now sees those applicants as
          perfectly average.
        </p>
        <Formula>{`income          → median-imputed value
income_missing  → 1

The model can now learn "missing income" as its own effect,
separately from "median income".`}</Formula>
        <p className="mt-2">
          This is why <Code>SimpleImputer(add_indicator=True)</Code> exists, and why adding the indicator is close
          to free insurance.
        </p>
      </Callout>

      <Callout variant="mistake">
        Computing the imputation statistic over the whole dataset. The median must come from the{" "}
        <strong>training fold only</strong> — otherwise test-set values influenced the value you filled training
        rows with. Same rule as scaling, same fix: put it in the pipeline.
      </Callout>
    </>
  );
}

/* ================================================================== */
/* 12 · Categorical Variables                                          */
/* ================================================================== */

export function Categorical() {
  return (
    <>
      <ConceptCard
        title="Encoding Categorical Variables"
        priority="must"
        whatItIs="Turning non-numeric categories into numbers, in a way that doesn't invent structure the categories don't have."
        intuition="The encoding is a claim about the categories. One-hot claims they're unrelated. Ordinal claims they're ordered and evenly spaced. Target encoding claims their relationship to the label is what matters. Pick the claim that's true."
        keyTakeaway="Unordered categories → one-hot (or the model's native handling). Ordered categories → ordinal. High cardinality → target encoding with out-of-fold means, or embeddings."
      />

      <Callout variant="bad" title="The classic trap — say this back to the interviewer">
        <Formula>{`colour:  Red = 1,  Green = 2,  Blue = 3`}</Formula>
        <p className="mt-2">
          You have just told the model that <strong>Green is greater than Red</strong>, that{" "}
          <strong>Blue is three times Red</strong>, and that <strong>Green is the midpoint of Red and Blue</strong>.
          A linear model will fit a single coefficient across that fake ordering. KNN and SVM will compute distances
          along it. None of that is true of colours.
        </p>
        <p className="mt-2">
          Trees survive it better — they can carve out <Code>colour ≤ 1.5</Code> and{" "}
          <Code>colour &gt; 2.5</Code> to isolate levels — but they still need more splits than they should, and
          the ordering constrains which groupings are reachable.
        </p>
      </Callout>

      <SubHeading priority="must">The five encodings</SubHeading>
      <ComparisonTable
        headers={["Encoding", "What it produces", "Use when", "Watch out for"]}
        rows={[
          [
            "One-Hot",
            "One binary column per level.",
            "Unordered categories with low cardinality (roughly under 15–50 levels).",
            "Explodes the feature count on high cardinality. Unseen levels at inference need a handling rule (handle_unknown='ignore').",
          ],
          [
            "Ordinal",
            "One integer column following a real ordering.",
            "Genuinely ordered levels: S < M < L, low < medium < high, bronze < silver < gold.",
            "Only valid when the order is real. It also asserts equal spacing, which may not hold.",
          ],
          [
            "Label",
            "One integer column, arbitrary order.",
            "Encoding the target of a classifier. That's essentially it.",
            "Using it on unordered input features is the mistake above. Tree models tolerate it; nothing else does.",
          ],
          [
            "Target / Mean",
            "Replace each level with the mean target for that level.",
            "High cardinality — zip code, merchant id, product sku — especially with GBMs.",
            "Leaks badly unless computed out-of-fold, and needs smoothing so rare levels don't get extreme values.",
          ],
          [
            "Embeddings",
            "A learned dense vector per level.",
            "Very high cardinality inside a neural network, or when levels have real semantic similarity.",
            "Needs a neural architecture and enough data per level to learn something.",
          ],
          [
            "Frequency / Count",
            "Replace the level with how often it occurs.",
            "A cheap, leak-free high-cardinality option; often a good baseline before target encoding.",
            "Two unrelated levels with the same frequency collide.",
          ],
        ]}
      />

      <Grid cols={2}>
        <DefinitionCard term="Target encoding, done safely">
          <Formula>{`# leaky — the row contributes to its own encoding
enc[c] = mean(y | category == c)

# safe — out-of-fold, plus smoothing toward the global mean
enc[c] = (n_c · mean_c + m · global_mean) / (n_c + m)
         computed on the OTHER folds only`}</Formula>
          <p className="mt-2">
            The smoothing term <Code>m</Code> pulls rare levels toward the global mean, so a category seen twice
            doesn't get a target mean of 1.0. Without out-of-fold computation, the encoding for a row includes that
            row's own label — direct target leakage, and validation scores will look wonderful.
          </p>
        </DefinitionCard>

        <DefinitionCard term="High cardinality: your options">
          <ul className="bullet-list">
            <li>
              <strong>Group the tail.</strong> Keep the top 20 levels, bucket everything else as "Other". Simple and
              surprisingly effective.
            </li>
            <li>
              <strong>Frequency encoding.</strong> One column, no leakage, no explosion.
            </li>
            <li>
              <strong>Out-of-fold target encoding.</strong> Strongest for GBMs, needs care.
            </li>
            <li>
              <strong>CatBoost.</strong> Its ordered target statistics do this correctly for you — that's the
              library's headline feature.
            </li>
            <li>
              <strong>Hashing trick.</strong> Hash levels into a fixed number of buckets. Collisions are the price of
              constant memory.
            </li>
            <li>
              <strong>Embeddings.</strong> If you're already in a neural network.
            </li>
          </ul>
        </DefinitionCard>
      </Grid>

      <Callout variant="tip" title="Two details that make you sound like you've shipped this">
        <ul className="bullet-list">
          <li>
            <strong>Unseen categories at inference.</strong> Production will hand you a level that wasn't in
            training. You need an explicit rule — an "unknown" bucket, or{" "}
            <Code>handle_unknown="ignore"</Code>. Silently crashing at 3am is the alternative.
          </li>
          <li>
            <strong>Dropping the first level.</strong> For linear models with an intercept, one-hot columns are
            perfectly collinear, so you drop one level as a reference — <Code>drop="first"</Code>. For trees and
            regularized models it doesn't matter, and dropping can even cost you a little.
          </li>
        </ul>
      </Callout>
    </>
  );
}

/* ================================================================== */
/* 13 · Class Imbalance                                                */
/* ================================================================== */

export function ClassImbalance() {
  return (
    <>
      <ConceptCard
        title="Class Imbalance"
        priority="must"
        whatItIs="When one class massively outnumbers the other, so a model can score well on aggregate metrics while being useless on the class you actually care about."
        intuition="The model isn't broken — it's doing exactly what you asked. If 99% of examples are negative, predicting 'negative' always is a genuinely excellent strategy for minimising error count. The problem is that error count was the wrong objective."
        example="1,000,000 transactions, 10,000 fraudulent (1%). Predict 'legitimate' for everything: 99% accuracy, 0% recall, £0 of fraud prevented."
        keyTakeaway="Imbalance is a metric problem first, a data problem second, and a threshold problem third. Fix the metric before you touch the data."
      />

      <Callout variant="warning" title="Fix the measurement before you fix the data">
        The instinct is to reach straight for SMOTE. Resist it. In order:
        <ol className="mt-2 space-y-1 pl-5 [&>li]:list-decimal [&>li]:text-muted">
          <li>
            <strong>Change the metric.</strong> PR-AUC, recall at a fixed precision, precision@K. Accuracy was
            never going to work.
          </li>
          <li>
            <strong>Tune the threshold.</strong> Free, instant, and often the entire fix. Nothing says the cutoff has
            to be 0.5.
          </li>
          <li>
            <strong>Use class weights.</strong> One argument, no synthetic data, keeps the true distribution.
          </li>
          <li>
            <strong>Only then resample</strong>, and only inside the training fold.
          </li>
        </ol>
      </Callout>

      <SubHeading priority="must">The toolbox</SubHeading>
      <ComparisonTable
        headers={["Technique", "What it does", "Pros", "Cons"]}
        rows={[
          [
            "Threshold tuning",
            "Move the decision cutoff away from 0.5 to hit a target precision or recall.",
            "Free, reversible, no retraining. Usually the first and best move.",
            "Doesn't improve the underlying ranking — it just chooses a different point on the same curve.",
          ],
          [
            "Class weights",
            "Weight the minority class higher in the loss (class_weight='balanced', scale_pos_weight).",
            "No synthetic data, no data thrown away, one argument.",
            "Distorts predicted probabilities — recalibrate if you need them to be meaningful.",
          ],
          [
            "Random undersampling",
            "Discard majority-class rows until the ratio improves.",
            "Fast training on huge datasets.",
            "Throws away real information. Poor choice unless the majority class is genuinely redundant.",
          ],
          [
            "Random oversampling",
            "Duplicate minority rows.",
            "Trivial to implement, keeps all data.",
            "Exact duplicates encourage memorisation of those specific rows.",
          ],
          [
            "SMOTE",
            "Generate synthetic minority points by interpolating between a minority point and its nearest minority neighbours.",
            "More variety than plain duplication; the standard reference answer.",
            "Can interpolate across a class boundary and manufacture nonsense. Weak in high dimensions and with categorical features. Must be inside the training fold.",
          ],
          [
            "Anomaly-detection framing",
            "Model the majority class and flag deviations.",
            "Works when positives are extremely rare or heterogeneous.",
            "Gives up the label information you do have.",
          ],
          [
            "Collect more positives",
            "Targeted labelling of likely positives, or a longer collection window.",
            "The only fix that adds real information.",
            "Slow and expensive — but say it, because it's the honest answer.",
          ],
        ]}
      />

      <Grid cols={2}>
        <DefinitionCard term="Threshold tuning, concretely">
          <p className="mb-2">
            Your classifier outputs a probability. 0.5 is a convention, not a law. Sweep it and pick the operating
            point your business actually wants:
          </p>
          <Formula>{`threshold ↑   →  fewer flags
                 precision ↑ (usually)
                 recall    ↓ (always)

threshold ↓   →  more flags
                 recall    ↑ (always)
                 precision ↓ (usually)`}</Formula>
          <p className="mt-2">
            Recall moves monotonically because lowering the bar can only add positives. Precision usually moves the
            other way but is <em>not</em> strictly monotone — it can wobble, especially in the sparse tail. Play with
            the threshold lab in the metrics section to see it.
          </p>
        </DefinitionCard>

        <Callout variant="mistake" title="SMOTE mistakes that get caught in interviews">
          <ul className="bullet-list">
            <li>
              <strong>Applying it before the split or before CV.</strong> Synthetic points derived from validation
              rows end up in training. Classic leakage; the scores look fantastic.
            </li>
            <li>
              <strong>Applying it to the validation or test set.</strong> Never. Evaluation must happen on the real
              distribution.
            </li>
            <li>
              <strong>Expecting it to beat class weights.</strong> On most tabular problems it doesn't. Benchmarks
              repeatedly find class weighting plus threshold tuning matches or beats SMOTE with less machinery.
            </li>
            <li>
              <strong>Using it on high-dimensional or categorical data.</strong> Interpolating between one-hot
              vectors produces points that correspond to nothing real.
            </li>
          </ul>
        </Callout>
      </Grid>

      <Callout variant="memorize">
        <div className="flex flex-wrap gap-2">
          <Chip tone="red">accuracy is out</Chip>
          <Chip tone="green">PR-AUC in</Chip>
          <Chip tone="green">tune the threshold first</Chip>
          <Chip tone="green">class_weight="balanced"</Chip>
          <Chip tone="amber">resample inside the fold only</Chip>
          <Chip tone="amber">recalibrate if you need probabilities</Chip>
        </div>
      </Callout>

      <SubHeading>Interview questions</SubHeading>
      <div className="space-y-2">
        <InterviewQuestion
          qa={{
            id: "ci1",
            q: "You have a dataset that's 99% negative. Walk me through your approach.",
            short: "Change the metric, tune the threshold, add class weights, and only then consider resampling — always inside the training fold.",
            strong:
              "First I'd stop using accuracy and switch to PR-AUC plus precision and recall at a chosen operating point, because at 1% positives accuracy tells me nothing. I'd use stratified splits and stratified CV so every fold keeps the ratio. Then I'd train a baseline with class_weight='balanced' — that's usually most of the benefit for one argument. Next I'd tune the threshold against the actual business constraint rather than 0.5; if analysts can review 200 cases a day, I'd optimise precision@200. Resampling like SMOTE I'd try last, strictly inside the training folds, and I'd expect it to help less than people assume. Finally, if I need the probabilities to be meaningful — for an expected-cost calculation — I'd recalibrate, because class weighting distorts them.",
          }}
        />
        <InterviewQuestion
          qa={{
            id: "ci2",
            q: "Does oversampling change the model's predicted probabilities?",
            short: "Yes. Resampling or class weighting shifts the implied base rate, so predicted probabilities come out systematically too high for the minority class.",
            strong:
              "Both resampling and class weights change the effective prior the model sees. If I balance a 1% positive rate to 50/50, the model learns as though positives were half the world, and its outputs are inflated accordingly. For ranking or a tuned threshold that's harmless — the ordering is preserved. But if a downstream system multiplies the probability by a dollar amount to get expected loss, those numbers are now wrong. The fixes are to recalibrate on an unresampled held-out set with Platt scaling or isotonic regression, or to adjust the intercept analytically for the known prior shift.",
          }}
        />
      </div>
    </>
  );
}
