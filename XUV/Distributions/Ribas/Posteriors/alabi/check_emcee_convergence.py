"""
Comprehensive convergence diagnostics for emcee MCMC chains.

This script checks:
1. Autocorrelation time convergence
2. Gelman-Rubin statistic (R-hat)
3. Trace plots for visual inspection
4. Acceptance fraction per walker
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import vplot

# Load the emcee samples
savedir = "gj1132_emcee/"
data = np.load(f"{savedir}/emcee_samples_final_custom_iter_401.npz")
chain = data['chain']  # Shape: (nwalkers, nsteps, ndim)
log_prob = data['log_prob']  # Shape: (nwalkers, nsteps)

nwalkers, nsteps, ndim = chain.shape
labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$", r"$t_{sat}$ [Gyr]",
          r"Age [Gyr]", r"$\beta_{XUV}$"]

print("="*70)
print("EMCEE CONVERGENCE DIAGNOSTICS")
print("="*70)
print(f"\nChain shape: {nwalkers} walkers × {nsteps} steps × {ndim} parameters")

# ===========================================================================
# 1. Autocorrelation time analysis
# ===========================================================================
print("\n" + "="*70)
print("1. AUTOCORRELATION TIME ANALYSIS")
print("="*70)

try:
    import emcee

    # Calculate autocorrelation time for each parameter
    tau = emcee.autocorr.integrated_time(chain, tol=0)

    print(f"\nAutocorrelation time (τ) per parameter:")
    for i, (label, t) in enumerate(zip(labels, tau)):
        print(f"  {label:25s}: τ = {t:7.2f} steps")

    print(f"\nMean autocorrelation time: {np.mean(tau):.2f} steps")
    print(f"Max autocorrelation time:  {np.max(tau):.2f} steps")

    # Rule of thumb: need at least 50× autocorrelation time
    min_steps_needed = 50 * np.max(tau)
    print(f"\nSteps per walker: {nsteps}")
    print(f"Minimum recommended: {min_steps_needed:.0f} steps (50 × max τ)")

    if nsteps >= min_steps_needed:
        print("✓ PASS: Chain length is sufficient")
    else:
        print(f"✗ WARNING: Need {min_steps_needed - nsteps:.0f} more steps")

except Exception as e:
    print(f"Could not calculate autocorrelation time: {e}")

# ===========================================================================
# 2. Gelman-Rubin statistic (R-hat)
# ===========================================================================
print("\n" + "="*70)
print("2. GELMAN-RUBIN STATISTIC (R-hat)")
print("="*70)
print("Compares variance within chains vs between chains")
print("R-hat ≈ 1.0 indicates convergence (< 1.1 is excellent, < 1.01 is ideal)")

def fbGelmanRubin(daChains):
    """
    Calculate Gelman-Rubin statistic for MCMC convergence.

    :param daChains: Chain data (nwalkers, nsteps, ndim)
    :returns: R-hat for each parameter
    """
    iNumChains, iNumSteps, iNumParams = daChains.shape

    daRhat = np.zeros(iNumParams)

    for i in range(iNumParams):
        # Chain means for each walker
        daChainMeans = np.mean(daChains[:, :, i], axis=1)
        # Overall mean
        dOverallMean = np.mean(daChainMeans)

        # Between-chain variance
        dB = (iNumSteps / (iNumChains - 1)) * np.sum((daChainMeans - dOverallMean)**2)

        # Within-chain variance
        daChainVars = np.var(daChains[:, :, i], axis=1, ddof=1)
        dW = np.mean(daChainVars)

        # Marginal posterior variance estimate
        dVarEstimate = ((iNumSteps - 1) / iNumSteps) * dW + (1 / iNumSteps) * dB

        # R-hat
        daRhat[i] = np.sqrt(dVarEstimate / dW)

    return daRhat

# Calculate R-hat using second half of chain (after burn-in)
burn = nsteps // 2
rhat = fbGelmanRubin(chain[:, burn:, :])

print(f"\nR-hat per parameter (using last {nsteps - burn} steps):")
for i, (label, r) in enumerate(zip(labels, rhat)):
    status = "✓" if r < 1.1 else "✗"
    print(f"  {status} {label:25s}: R-hat = {r:.6f}")

print(f"\nMax R-hat: {np.max(rhat):.6f}")
if np.all(rhat < 1.01):
    print("✓ EXCELLENT: All R-hat < 1.01 (chains have converged)")
elif np.all(rhat < 1.1):
    print("✓ GOOD: All R-hat < 1.1 (chains likely converged)")
else:
    print("✗ WARNING: Some R-hat ≥ 1.1 (chains may not have converged)")

# ===========================================================================
# 3. Acceptance fraction per walker
# ===========================================================================
print("\n" + "="*70)
print("3. ACCEPTANCE FRACTION ANALYSIS")
print("="*70)
print("Healthy range: 0.2 - 0.5")

# Calculate acceptance fraction by checking when log_prob changes
accept_frac = np.zeros(nwalkers)
for i in range(nwalkers):
    # Count steps where log_prob changed (accepted proposal)
    accept_frac[i] = np.sum(np.diff(log_prob[i, :]) != 0) / (nsteps - 1)

print(f"\nMean acceptance fraction: {np.mean(accept_frac):.3f}")
print(f"Min acceptance fraction:  {np.min(accept_frac):.3f}")
print(f"Max acceptance fraction:  {np.max(accept_frac):.3f}")
print(f"Std acceptance fraction:  {np.std(accept_frac):.3f}")

if np.mean(accept_frac) < 0.2:
    print("✗ WARNING: Low acceptance (<0.2). Consider decreasing step size.")
elif np.mean(accept_frac) > 0.5:
    print("✗ WARNING: High acceptance (>0.5). Consider increasing step size.")
else:
    print("✓ PASS: Acceptance fraction in healthy range")

# ===========================================================================
# 4. Visual diagnostics
# ===========================================================================
print("\n" + "="*70)
print("4. GENERATING VISUAL DIAGNOSTICS")
print("="*70)

# Trace plots
fig, axes = plt.subplots(ndim, 1, figsize=(12, 2*ndim), sharex=True)

for i in range(ndim):
    ax = axes[i]
    # Plot a subset of walkers for clarity (every 10th walker)
    for j in range(0, nwalkers, 10):
        ax.plot(chain[j, :, i], alpha=0.3, lw=0.5, color='C0')

    ax.set_ylabel(labels[i], fontsize=12)
    ax.axvline(burn, color='r', linestyle='--', alpha=0.5, label='Burn-in' if i == 0 else '')

    if i == 0:
        ax.legend(loc='upper right')

axes[-1].set_xlabel("Step number", fontsize=12)
fig.suptitle("MCMC Trace Plots (every 10th walker shown)", fontsize=14, y=0.995)
fig.tight_layout()
fig.savefig(f"{savedir}/convergence_trace_plots.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved trace plots: {savedir}/convergence_trace_plots.png")

# Autocorrelation plots
try:
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2*ndim), sharex=True)

    for i in range(ndim):
        ax = axes[i]

        # Calculate autocorrelation for this parameter (average over walkers)
        max_lag = min(500, nsteps // 2)
        autocorr = np.zeros(max_lag)

        for lag in range(max_lag):
            c0 = np.mean([np.var(chain[j, :, i]) for j in range(nwalkers)])
            ct = np.mean([np.mean((chain[j, :-lag or None, i] - np.mean(chain[j, :, i])) *
                                  (chain[j, lag:, i] - np.mean(chain[j, :, i])))
                         for j in range(nwalkers)])
            autocorr[lag] = ct / c0

        ax.plot(autocorr, 'C0')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(1/np.e, color='r', linestyle='--', alpha=0.5,
                  label='1/e threshold' if i == 0 else '')
        ax.set_ylabel(f"{labels[i]}\nAutocorr", fontsize=10)
        ax.set_ylim(-0.1, 1.1)

        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel("Lag (steps)", fontsize=12)
    fig.suptitle("Autocorrelation Functions", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(f"{savedir}/convergence_autocorr.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved autocorrelation plots: {savedir}/convergence_autocorr.png")

except Exception as e:
    print(f"Could not create autocorrelation plots: {e}")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "="*70)
print("CONVERGENCE SUMMARY")
print("="*70)

bConverged = True
listIssues = []

if nsteps < min_steps_needed:
    bConverged = False
    listIssues.append(f"Chain too short (need {min_steps_needed:.0f} steps)")

if np.any(rhat >= 1.1):
    bConverged = False
    listIssues.append(f"High R-hat (max={np.max(rhat):.4f} ≥ 1.1)")

if np.mean(accept_frac) < 0.2 or np.mean(accept_frac) > 0.5:
    listIssues.append(f"Suboptimal acceptance fraction ({np.mean(accept_frac):.3f})")

if bConverged:
    print("\n✓✓✓ CHAINS HAVE CONVERGED ✓✓✓")
    print("\nAll convergence criteria passed. Posterior samples are reliable.")
else:
    print("\n✗✗✗ CONVERGENCE ISSUES DETECTED ✗✗✗")
    print("\nIssues found:")
    for issue in listIssues:
        print(f"  - {issue}")
    print("\nRecommendations:")
    if nsteps < min_steps_needed:
        print(f"  1. Run longer chains ({min_steps_needed:.0f} steps recommended)")
    if np.any(rhat >= 1.1):
        print("  2. Check trace plots for non-stationarity")
        print("  3. Consider longer burn-in period")

print("\n" + "="*70)
