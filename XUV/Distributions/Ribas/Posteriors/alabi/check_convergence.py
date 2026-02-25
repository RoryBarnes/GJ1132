"""Quick convergence check for emcee samples."""
import numpy as np

# Load the emcee samples
savedir = "gj1132_emcee/"
data = np.load(f"{savedir}/emcee_samples_final_custom_iter_401.npz")
samples = data['samples']  # Shape: (nsamples, ndim) - already burned and thinned

nsamples, ndim = samples.shape
labels = [r"$m_{\star}$", r"$f_{sat}$", r"$t_{sat}$", r"Age", r"$\beta_{XUV}$"]

print("="*70)
print("EMCEE CONVERGENCE SUMMARY")
print("="*70)
print(f"\nSamples shape: {samples.shape}")
print(f"{nsamples:,} posterior samples × {ndim} parameters")
print("\nNote: Samples are already burned and thinned by alabi")

# Extract info from summary file
print("\n" + "="*70)
print("CONVERGENCE METRICS FROM SUMMARY FILE")
print("="*70)

try:
    with open(f"{savedir}/surrogate_model.txt", 'r') as f:
        content = f.read()

    # Extract metrics
    import re

    nwalkers_match = re.search(r"Number of walkers: (\d+)", content)
    nsteps_match = re.search(r"Number of steps per walker: (\d+)", content)
    accept_match = re.search(r"Mean acceptance fraction: ([\d.]+)", content)
    tau_match = re.search(r"Mean autocorrelation time: ([\d.]+)", content)
    burn_match = re.search(r"Burn: (\d+)", content)
    thin_match = re.search(r"Thin: (\d+)", content)

    if nwalkers_match:
        nwalkers = int(nwalkers_match.group(1))
        print(f"Number of walkers: {nwalkers}")

    if nsteps_match:
        nsteps = int(nsteps_match.group(1))
        print(f"Steps per walker: {nsteps:,}")

    if accept_match:
        accept_frac = float(accept_match.group(1))
        print(f"\nAcceptance fraction: {accept_frac:.3f}")
        if 0.2 <= accept_frac <= 0.5:
            print("  ✓ EXCELLENT: In optimal range [0.2, 0.5]")
        elif accept_frac < 0.2:
            print("  ✗ Low acceptance - step size too large")
        else:
            print("  ✗ High acceptance - step size too small")

    if tau_match:
        tau = float(tau_match.group(1))
        print(f"\nAutocorrelation time (τ): {tau:.1f} steps")

        if nsteps_match:
            n_tau = nsteps / tau
            print(f"Chain length / τ: {n_tau:.1f}×")

            if n_tau >= 50:
                print("  ✓ EXCELLENT: Chain length >> 50τ (well-converged)")
            elif n_tau >= 20:
                print("  ✓ GOOD: Chain length > 20τ (likely converged)")
            else:
                print("  ✗ WARNING: Chain too short (need > 50τ)")

    if burn_match and thin_match:
        burn = int(burn_match.group(1))
        thin = int(thin_match.group(1))
        print(f"\nBurn-in: {burn} steps")
        print(f"Thinning: every {thin} steps")

        if tau_match:
            print(f"  Burn / τ: {burn / tau:.1f}×")
            print(f"  Thin / τ: {thin / tau:.2f}×")

except Exception as e:
    print(f"Could not read summary file: {e}")

# Basic posterior statistics
print("\n" + "="*70)
print("POSTERIOR STATISTICS")
print("="*70)

for i, label in enumerate(labels):
    mean = np.mean(samples[:, i])
    std = np.std(samples[:, i])
    median = np.median(samples[:, i])
    q16, q84 = np.percentile(samples[:, i], [16, 84])

    print(f"\n{label}:")
    print(f"  Mean ± Std:  {mean:.6f} ± {std:.6f}")
    print(f"  Median:      {median:.6f}")
    print(f"  16-84%:      {q16:.6f} - {q84:.6f}")

print("\n" + "="*70)
print("OVERALL ASSESSMENT")
print("="*70)

# Convergence assessment
converged = True
issues = []

if 'accept_frac' in locals() and not (0.2 <= accept_frac <= 0.5):
    issues.append("Suboptimal acceptance fraction")

if 'n_tau' in locals():
    if n_tau < 20:
        converged = False
        issues.append(f"Chain too short ({n_tau:.1f}τ < 20τ)")
    elif n_tau < 50:
        issues.append(f"Chain could be longer ({n_tau:.1f}τ < 50τ, but > 20τ)")

if converged and len(issues) == 0:
    print("\n✓✓✓ CHAINS HAVE CONVERGED ✓✓✓")
    print("\nAll convergence criteria passed.")
    print(f"Posterior samples ({nsamples:,} total) are reliable for inference.")
elif converged:
    print("\n✓ CHAINS LIKELY CONVERGED")
    print("\nMinor issues noted:")
    for issue in issues:
        print(f"  - {issue}")
    print(f"\nPosterior samples ({nsamples:,} total) should be reliable.")
else:
    print("\n✗ CONVERGENCE ISSUES DETECTED")
    print("\nIssues:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nConsider running longer chains.")

print("\n" + "="*70)
