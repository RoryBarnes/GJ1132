# Hybrid Active Learning Strategy

## Motivation

During active learning, alabi runs one VPLanet call per iteration (chosen by the
acquisition function) while N-1 CPU cores sit idle. The idea is to fill those
idle cores with random (space-filling) VPLanet evaluations that are added to the
GP training set alongside the acquisition-chosen point. This yields N points per
iteration at the wall-clock cost of 1.

## Why It Works

1. The GP conditions on all training points equally regardless of how they were
   selected. Random points improve global coverage; the acquisition point refines
   the posterior-relevant region. Both are valid training data.

2. The VPLanet calls are embarrassingly parallel and fast (~0.75s each on this
   machine). The dominant cost per iteration is GP hyperparameter optimization,
   which scales with the total number of training points and happens every
   `gp_opt_freq` iterations.

3. Early stopping on the test NRMSE (already computed at every iteration by
   alabi at negligible cost) avoids wasting iterations once the GP is accurate
   enough. The target threshold is 1%.

## Timing Estimates (9-core machine, GJ 1132 5D problem)

Assumes the GP needs ~3000 total training points to reach the 1% NRMSE target.

### Brute Force: ninit=3000, niter=500, gp_opt_freq=10

| Component        | Time     |
|------------------|----------|
| Initial samples  | 5 min    |
| Grid search      | 25 min   |
| Active learning  | 265 min  |
| Samplers         | 20 min   |
| **Total**        | **315 min (5.3 hrs)** |

Active learning dominates: 50 HP optimizations at ~4 min each = 200 min, plus
65 min for 490 regular iterations at ~8 s each.

### Hybrid: ninit=1500, niter~190, 8 points/iter, gp_opt_freq=10

| Component        | Time     |
|------------------|----------|
| Initial samples  | 2 min    |
| Grid search      | 10 min   |
| Active learning  | 71 min   |
| Samplers         | 20 min   |
| **Total**        | **104 min (1.7 hrs)** |

Each iteration adds 8 points (1 acquisition + 7 random) in parallel. After ~190
iterations the training set reaches ~3020 points. Only 19 HP optimizations are
needed vs 50 in the brute force case.

**Estimated speedup: ~3x** (savings of ~210 min).

Note: the grid search time depends on ninit, not the active learning strategy,
so it is shorter here only because ninit is smaller. For a fair comparison at
fixed ninit=1500, the grid search cost is the same in both approaches.

## Optimization Space

Three parameters to tune:
- `ninit_train`: initial space-filling sample size
- `niter`: maximum active learning iterations (with early stopping)
- `n_bonus`: number of random points per iteration (up to N_cores - 1)

The key tradeoff: more initial points give a better GP foundation and faster
grid search convergence, but fewer active learning iterations are needed. The
hybrid approach lets you start with fewer initial points and grow the training
set in parallel batches during active learning, minimizing the total number of
HP optimizations.

## Implementation Notes

This requires modifying `alabi.core.SurrogateModel.active_train()` to:
1. Accept an `n_bonus` parameter
2. Generate `n_bonus` random points from the prior at each iteration
3. Run all VPLanet calls (1 acquisition + n_bonus random) in parallel
4. Add all results to the training set before the next GP update
5. Check the test NRMSE against a threshold and halt if satisfied

The NRMSE check is trivial: alabi already computes `test_scaled_mse` at every
iteration, and `NRMSE = 100 * sqrt(test_scaled_mse)`.

## Evidence From This Investigation

- ninit=500, niter=500: severe overfitting (test NRMSE 9% -> 97%)
- ninit=1500, niter=500: no overfitting, test NRMSE plateaus at 2.73%
- ninit=3000, niter=500: in progress (testing whether more points break 1%)

The pattern confirms that global coverage (space-filling points) is more
valuable than targeted acquisition for reducing test error on this problem.
The hybrid approach provides both simultaneously.
