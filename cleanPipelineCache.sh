#!/usr/bin/env bash
# cleanPipelineCache.sh -- Delete all cached/stale intermediate files
# so director.py regenerates everything from scratch.
#
# TRUE INPUTS (never deleted): ensemble_FFD.csv, *.in files, FFD.py,
# gj1132_ribas.json, MagmaOceanFinal.npy, flare_mcmc_samples_fixed_slope.npy,
# data/tess_cache/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Pipeline cache cleaner for GJ 1132"
echo "Repo root: ${REPO_ROOT}"
echo ""

# Scene 01: Kepler FFD MCMC cache
echo "Scene 01: Kepler FFD..."
rm -f "${REPO_ROOT}/XUV/KeplerFlares/flare_mcmc_samples.npy"
rm -f "${REPO_ROOT}/XUV/KeplerFlares/flare_mcmc_samples.txt"
rm -f "${REPO_ROOT}/XUV/KeplerFlares/flare_mcmc_samples.csv"
rm -f "${REPO_ROOT}/XUV/KeplerFlares/flare_mcmc_samples_param_names.txt"
rm -f "${REPO_ROOT}/XUV/KeplerFlares/kepler_ffd_posterior_stats.json"

# Scene 03: flare_candidates.json is a TRUE INPUT (human-labeled),
# not a pipeline intermediate. Do NOT delete it.

# Scene 06: LXUV Monte Carlo samples
echo "Scene 06: LXUV samples..."
rm -f "${REPO_ROOT}/XUV/Distributions/Ribas/LXUV/lxuv_samples.txt"

# Scene 07: Engle age samples
echo "Scene 07: Engle age samples..."
rm -f "${REPO_ROOT}/XUV/Distributions/Engle/Age/age_samples.txt"

# Scene 08: alabi output directory (surrogate, samplers, grid search)
echo "Scene 08: alabi output..."
ALABI_OUTPUT="${REPO_ROOT}/XUV/Distributions/Ribas/Posteriors/alabi/output"
rm -f "${ALABI_OUTPUT}/surrogate_model.pkl"
rm -f "${ALABI_OUTPUT}/emcee_samples.npz"
rm -f "${ALABI_OUTPUT}/dynesty_samples.npz"
rm -f "${ALABI_OUTPUT}/multinest_samples.npz"
rm -f "${ALABI_OUTPUT}/ultranest_samples.npz"
rm -f "${ALABI_OUTPUT}/gp_grid_search_results.txt"
rm -f "${ALABI_OUTPUT}/dynesty_transform_final.npy"
rm -f ${ALABI_OUTPUT}/initial_*_sample.npz 2>/dev/null || true

# Scene 08 setup: MaxLEV output
echo "Scene 08: MaxLEV output..."
rm -f "${REPO_ROOT}/XUV/EvolutionPlots/MaximumLikelihood/maxlike_results.txt"

# Stale priors directory copies
echo "Stale priors..."
rm -f "${REPO_ROOT}/priors/dynesty_transform_final.npy"
rm -f "${REPO_ROOT}/priors/age_samples.txt"

# Scenes 09-11: vconverge output directories
CUMXUV="${REPO_ROOT}/XUV/Distributions/CumulativeXUV"
for DIR_NAME in EngleBarnes RibasBarnes Engle Ribas \
                EngleModelErrorsOnly EngleStellarErrorsOnly \
                RibasModelErrorsOnly RibasStellarErrorsOnly; do
    echo "vconverge: ${DIR_NAME}..."
    rm -f "${CUMXUV}/${DIR_NAME}/output/Converged_Param_Dictionary.json"
    rm -rf ${CUMXUV}/${DIR_NAME}/output/xuv_* 2>/dev/null || true
    # Remove copied prior files
    rm -f "${CUMXUV}/${DIR_NAME}/flares_variable_slope.npy"
    rm -f "${CUMXUV}/${DIR_NAME}/age_samples.txt"
    rm -f "${CUMXUV}/${DIR_NAME}/dynesty_transform_final.npy"
done

# Scene 10: CosmicShoreline vplanet output
echo "Scene 10: CosmicShoreline vplanet output..."
rm -f "${REPO_ROOT}/XUV/CosmicShoreline/solarsystem.log"
rm -f ${REPO_ROOT}/XUV/CosmicShoreline/*.forward 2>/dev/null || true

# All plot outputs
echo "Plot outputs..."
rm -rf "${REPO_ROOT}/Plot" 2>/dev/null || true

echo ""
echo "Done. All intermediate files removed."
echo "Run 'python director.py' to regenerate from scratch."
