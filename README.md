# Machine Learning Fundamentals — 4-Hour AI Engineer Interview Review

A self-contained, interactive review site for the ML fundamentals portion of an
**AI Engineer / ML Engineer** interview. Built for one focused four-hour pass:
intuition, the formula worth memorising, the sentence you say out loud, and the
mistake that gives you away.

## Run it

```bash
npm install
```

```bash
npm run dev
```

Then open the URL Vite prints (usually `http://localhost:5173`).

```bash
npm run build
```

Builds to `dist/` — a static bundle you can open from any static host. There is
no backend; all progress lives in `localStorage`.

## What's in it

**25 sections** in a fixed order, split across four one-hour blocks:

| Hour | Focus | Sections |
| --- | --- | --- |
| 1 | Core concepts | Problem types · Train/val/test · Data leakage · Over/underfitting · Bias–variance · Cross-validation |
| 2 | Metrics & data prep | Evaluation metrics · Feature engineering · Scaling · Missing data · Categoricals · Class imbalance |
| 3 | Models & training | Regularization · Gradient descent · Parameters vs hyperparameters · 8 algorithm cards + comparison table |
| 4 | Interview review | Model selection · Ensembles · PCA · Clustering · Pipeline · 47 Q&As · 16 scenarios · 82-question quiz · Cheat sheet |

Every topic follows the same shape where it helps: **what it is → intuition →
example → the interview answer → the common mistake → the key takeaway.**

### Interactive labs

These compute real numbers from real (seeded, deterministic) data — nothing is
hand-drawn:

- **Bias–Variance Lab** — slider over model complexity, live bias²/variance/total error.
- **Overfitting Lab** — genuine polynomial least-squares fit on 14 train / 14 validation
  points, degree 1–12, with the train and validation error curves.
- **Threshold / Confusion-Matrix Lab** — 1,000 cases at a 3% positive rate. Move the
  threshold and watch the confusion matrix, precision, recall, F1, specificity and
  accuracy move with it, alongside live ROC and PR curves.
- **Regularization Lab** — L1 vs L2 vs Elastic Net coefficient paths as λ rises.
- **Gradient Descent Lab** — step or animate the descent; learning-rate presets show the
  slow, healthy, oscillating and divergent regimes exactly.
- **K-Means Lab** — step through assign/update, re-seed to see initialisation sensitivity,
  with an elbow plot computed from multi-restart runs.
- **PCA Lab** — toggle standardization and watch PC1 stop being "whichever feature has the
  biggest units"; project onto PC1 to see the reconstruction error.
- **K-Fold diagram**, **split-ratio picker**, **sigmoid curve**, **ensemble diagrams**.

### Features

- Sticky sidebar with search, per-section time estimates and priority dots.
- Priority labels: 🔴 Must Know · 🟠 Important · 🟢 Good to Know.
- Per-section completion checkboxes; progress, time remaining and "next up" in the header.
- Collapsible interview answers with three depths: short answer → what to say → deeper.
- Rapid-fire quiz with self-grading, per-topic filtering, best-score tracking and a
  "drill the misses" round.
- Dark mode, keyboard-navigable, responsive from 375px up.
- Cheat sheet is print-styled — `⎙ Print / save as PDF` gives you a clean one-pager.
- Interview readiness checklist at the end.

All state (section progress, quiz best score, readiness checklist) is namespaced under
`mlprep.` in `localStorage`. **Reset all progress** at the bottom of the sidebar clears it.

## Project layout

```
src/
  data/           section metadata, interview Q&As, scenarios, quiz bank
  sections/       one module per hour + metrics, algorithms, questions, quiz, cheat sheet
  components/     TopicSection, ConceptCard, InterviewQuestion, QuizCard,
                  ComparisonTable, FormulaCard, Callout, Sidebar, AlgorithmCard
  components/viz/ the interactive labs and diagrams (hand-rolled SVG, no chart library)
  hooks/          localStorage-backed progress, theme, active-section tracking
```

Stack: React 19 + TypeScript + Tailwind CSS v4 + Vite. Three runtime dependencies
(`react`, `react-dom`) and no charting, animation or UI libraries.
