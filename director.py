#!/usr/bin/env python3
"""
Orchestrate generation of all figures for the GJ 1132 XUV paper.

Reads constrain.json for per-figure configuration and generates each
figure in the correct order, placing output in the Figures/ directory.

Usage:
    python director.py --config constrain.json
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import contextlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

KEPLER_FLARES_DIR = os.path.join(REPO_ROOT, "XUV", "KeplerFlares")
TESS_DIR = os.path.join(REPO_ROOT, "XUV", "TESS")
LXUV_DIR = os.path.join(REPO_ROOT, "XUV", "Distributions", "Ribas", "LXUV")
POSTERIORS_DIR = os.path.join(
    REPO_ROOT, "XUV", "Distributions", "Ribas", "Posteriors", "alabi"
)
ENGLE_AGE_DIR = os.path.join(
    REPO_ROOT, "XUV", "Distributions", "Engle", "Age"
)
CUMULATIVE_XUV_DIR = os.path.join(
    REPO_ROOT, "XUV", "Distributions", "CumulativeXUV"
)
COSMIC_SHORELINE_DIR = os.path.join(REPO_ROOT, "XUV", "CosmicShoreline")
MAXLEV_DIR = os.path.join(REPO_ROOT, "XUV", "EvolutionPlots", "MaximumLikelihood")
VPLANET_NATIVE_BIN_DIR = "/workspace/vplanet-private/bin"

VALID_MODES = {"plot-only", "precomputed", "full", "standards"}

# Map of figure key -> list of expected output base names (without extension).
# Extension is determined by mode via fsResolveOutputExtension(), except for
# figures whose generators always produce PDF regardless of mode.
EXPECTED_FILES = {
    "figure01": ["CornerVariableSlope"],
    "figure02": ["ffd_comp"],
    "figure03": ["GJ1132_flares"],
    "figure04": ["GJ1132_FFD_comp2"],
    "figure05": ["lxuv_hist", "log_lxuv_lbol_hist"],
    "figure06": ["sampler_comparison"],
    "figure07": ["EngleAgeHist"],
    "figure08": ["XUVEvol"],
    "figure09": ["GJ1132b_CumulativeXUV_Multi", "CosmicShoreline"],
    "figure10": ["GJ1132b_ErrorSourceComparison"],
    "figure11": ["FitComparison"],
}

# Figures whose internal generators always produce PDF output.
FIXED_PDF_FIGURES = {"figure03", "figure04", "figure05", "figure07", "figure08", "figure11"}


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def fcontextWorkingDirectory(sPath):
    """Temporarily change into *sPath*, then restore the original cwd."""
    sOriginal = os.getcwd()
    os.chdir(sPath)
    try:
        yield
    finally:
        os.chdir(sOriginal)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def fdictLoadConfiguration(sConfigPath):
    """Load *sConfigPath* and return the parsed dictionary."""
    with open(sConfigPath, "r") as fileHandle:
        dictConfig = json.load(fileHandle)
    if not fbValidateConfiguration(dictConfig):
        print("ERROR: Invalid configuration file.")
        sys.exit(1)
    return dictConfig


def fbValidateConfiguration(dictConfig):
    """Return True when all required keys and mode values are present."""
    listRequired = ["sOutputDirectory", "dictFigures"]
    for sKey in listRequired:
        if sKey not in dictConfig:
            print(f"Missing required key: {sKey}")
            return False
    for sKey, dictFigure in dictConfig["dictFigures"].items():
        sMode = dictFigure.get("sMode", "")
        if sMode not in VALID_MODES:
            print(f"{sKey}: invalid sMode '{sMode}'")
            return False
    return True


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def fsResolveOutputExtension(sMode):
    """Return the file extension for the given mode."""
    if sMode == "standards":
        return ".png"
    return ".pdf"


def fsResolveOutputPath(sOutputDirectory, sFilename):
    """Return the full path inside *sOutputDirectory* for *sFilename*."""
    sDir = os.path.join(REPO_ROOT, sOutputDirectory)
    os.makedirs(sDir, exist_ok=True)
    return os.path.join(sDir, sFilename)


def fiDetermineNumberOfCores(iRequestedCores):
    """Return the actual core count; -1 means total minus one."""
    iTotal = multiprocessing.cpu_count()
    if iRequestedCores == -1:
        return max(1, iTotal - 1)
    return min(iRequestedCores, iTotal)


def fnCopyFigureFile(sSourcePath, sDestinationPath):
    """Copy *sSourcePath* to *sDestinationPath* with a status message."""
    if not os.path.exists(sSourcePath):
        raise FileNotFoundError(f"Source file not found: {sSourcePath}")
    os.makedirs(os.path.dirname(sDestinationPath), exist_ok=True)
    shutil.copy2(sSourcePath, sDestinationPath)
    print(f"  Copied -> {sDestinationPath}")


def fsLocateExecutable(sName):
    """Return the full path to *sName*, searching PATH and common locations."""
    sFound = shutil.which(sName)
    if sFound is not None:
        return sFound
    for sCandidate in [
        os.path.join(os.path.dirname(sys.executable), sName),
        os.path.join(os.path.expanduser("~"), ".local", "bin", sName),
        os.path.join("/usr/local/bin", sName),
    ]:
        if os.path.isfile(sCandidate):
            return sCandidate
    raise FileNotFoundError(f"Cannot find executable: {sName}")


def fnRunSubprocess(sScriptName, sWorkingDirectory, listArgs=None):
    """Execute *sScriptName* in *sWorkingDirectory* via subprocess."""
    listCommand = [sys.executable, sScriptName]
    if listArgs:
        listCommand.extend(listArgs)
    result = subprocess.run(
        listCommand,
        cwd=sWorkingDirectory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"{sScriptName} exited with code {result.returncode}")
    return result


# ---------------------------------------------------------------------------
# Shared TESS analysis
# ---------------------------------------------------------------------------

_dictTessCache = {}


def fdictRunTessAnalysis(dictGlobalConfig):
    """Run the TESS flare pipeline once and cache all intermediate results."""
    if _dictTessCache:
        return _dictTessCache

    sCacheDir = dictGlobalConfig.get("sTessDataCache", "data/tess_cache")
    os.environ["LIGHTKURVE_CACHE_DIR"] = os.path.abspath(
        os.path.join(REPO_ROOT, sCacheDir)
    )

    sys.path.insert(0, TESS_DIR)
    from flares_from_TESS import (
        calculate_total_exposure,
        compute_and_fit_ffd,
        compute_flare_equivalent_durations,
        download_tess_data,
        get_cluster_data,
        get_flare_parameters,
        get_literature_comparison_data,
    )

    lightcurve = download_tess_data()
    dTotalExposure = calculate_total_exposure(lightcurve)
    sectors, tStart, tStop, lumin = get_flare_parameters()
    daEquivDurations = compute_flare_equivalent_durations(
        lightcurve, sectors, tStart, tStop
    )
    ffdX, ffdY, ffdXerr, ffdYerr, dictFitResults = compute_and_fit_ffd(
        daEquivDurations, dTotalExposure, lumin, lightcurve, tStart, tStop
    )
    dictLiterature = get_literature_comparison_data()
    dictClusters = get_cluster_data()

    _dictTessCache.update({
        "lightcurve": lightcurve,
        "sectors": sectors,
        "tStart": tStart,
        "tStop": tStop,
        "lumin": lumin,
        "daEquivDurations": daEquivDurations,
        "dTotalExposure": dTotalExposure,
        "ffdX": ffdX,
        "ffdY": ffdY,
        "ffdXerr": ffdXerr,
        "ffdYerr": ffdYerr,
        "dictFitResults": dictFitResults,
        "dictLiterature": dictLiterature,
        "dictClusters": dictClusters,
    })
    return _dictTessCache


# ---------------------------------------------------------------------------
# Per-figure generators
# ---------------------------------------------------------------------------


def fnGenerateFigure01(dictFigConfig, dictGlobalConfig):
    """Figure 1: CornerVariableSlope — Kepler FFD corner plot."""
    sMode = dictFigConfig["sMode"]
    sOutDir = dictGlobalConfig["sOutputDirectory"]
    sExtension = fsResolveOutputExtension(sMode)
    sOutputFile = fsResolveOutputPath(sOutDir, "CornerVariableSlope" + sExtension)

    if sMode in ("full", "standards"):
        sGeneratedFile = os.path.join(
            KEPLER_FLARES_DIR, "CornerVariableSlope" + sExtension
        )
        fnRunSubprocess(
            "kepler_ffd.py", KEPLER_FLARES_DIR, [sGeneratedFile]
        )
        fnCopyFigureFile(sGeneratedFile, sOutputFile)
        return

    # plot-only or precomputed: build corner from saved samples
    import corner

    sSamplesPath = os.path.join(KEPLER_FLARES_DIR, "flare_mcmc_samples.npy")
    daSamples = np.load(sSamplesPath)
    fig = corner.corner(
        daSamples,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    plt.savefig(sOutputFile, dpi=150, bbox_inches="tight")
    plt.close()


def fnGenerateFigure02(dictFigConfig, dictGlobalConfig):
    """Figure 2: ffd_comp — FFD comparison at four ages."""
    sMode = dictFigConfig["sMode"]
    sExtension = fsResolveOutputExtension(sMode)
    sOutputFile = fsResolveOutputPath(
        dictGlobalConfig["sOutputDirectory"], "ffd_comp" + sExtension
    )

    sSamplesPath = os.path.join(KEPLER_FLARES_DIR, "flare_mcmc_samples.npy")
    daSamples = np.load(sSamplesPath)
    listMedianParams = [np.median(daSamples[:, i]) for i in range(6)]

    with fcontextWorkingDirectory(KEPLER_FLARES_DIR):
        sys.path.insert(0, KEPLER_FLARES_DIR)
        from ffd_age import Plot

        Plot(listMedianParams, mass=0.5, filename=sOutputFile)
    plt.close("all")


def fnGenerateFigure03(dictFigConfig, dictGlobalConfig):
    """Figure 3: GJ1132_flares — three TESS flare lightcurves."""
    sMode = dictFigConfig["sMode"]
    sExtension = fsResolveOutputExtension(sMode)
    sOutputFile = fsResolveOutputPath(
        dictGlobalConfig["sOutputDirectory"], "GJ1132_flares" + sExtension
    )
    dictTess = fdictRunTessAnalysis(dictGlobalConfig)

    sys.path.insert(0, TESS_DIR)
    from flares_from_TESS import plot_flare_lightcurves

    with fcontextWorkingDirectory(TESS_DIR):
        plot_flare_lightcurves(
            dictTess["lightcurve"],
            dictTess["sectors"],
            dictTess["tStart"],
            dictTess["tStop"],
        )
    fnCopyFigureFile(
        os.path.join(TESS_DIR, "GJ1132_flares.pdf"), sOutputFile
    )
    plt.close("all")


def fnGenerateFigure04(dictFigConfig, dictGlobalConfig):
    """Figure 4: GJ1132_FFD_comp2.pdf — comprehensive FFD comparison."""
    sOutputFile = fsResolveOutputPath(
        dictGlobalConfig["sOutputDirectory"], "GJ1132_FFD_comp2.pdf"
    )
    dictTess = fdictRunTessAnalysis(dictGlobalConfig)

    sys.path.insert(0, TESS_DIR)
    from flares_from_TESS import plot_comprehensive_comparison

    dAlpha = dictTess["dictFitResults"]["alpha"]
    dBeta = dictTess["dictFitResults"]["beta"]

    with fcontextWorkingDirectory(TESS_DIR):
        plot_comprehensive_comparison(
            dictTess["ffdX"],
            dictTess["ffdY"],
            dictTess["ffdYerr"],
            dAlpha,
            dBeta,
            dictTess["dictLiterature"],
            dictTess["dictClusters"],
        )
    fnCopyFigureFile(
        os.path.join(TESS_DIR, "GJ1132_FFD_comp2.pdf"), sOutputFile
    )
    plt.close("all")


def fnGenerateFigure05(dictFigConfig, dictGlobalConfig):
    """Figure 5: lxuv_hist.pdf + log_lxuv_lbol_hist.pdf."""
    sOutDir = dictGlobalConfig["sOutputDirectory"]
    with fcontextWorkingDirectory(LXUV_DIR):
        sys.path.insert(0, LXUV_DIR)
        from lxuv import main as fnLxuvMain

        fnLxuvMain()

    fnCopyFigureFile(
        os.path.join(LXUV_DIR, "lxuv_hist.pdf"),
        fsResolveOutputPath(sOutDir, "lxuv_hist.pdf"),
    )
    fnCopyFigureFile(
        os.path.join(LXUV_DIR, "log_lxuv_lbol_hist.pdf"),
        fsResolveOutputPath(sOutDir, "log_lxuv_lbol_hist.pdf"),
    )
    plt.close("all")


def fnEnsureMaxLevResults():
    """Run MaxLEV MAP estimation if results file does not already exist."""
    sResultsFile = os.path.join(MAXLEV_DIR, "maxlike_results.txt")
    if os.path.exists(sResultsFile):
        print(f"  MaxLEV results already exist: {sResultsFile}")
        return
    print("  Running MaxLEV for MAP estimation...")
    dictEnv = fnBuildNativeBinaryEnvironment()
    subprocess.run(
        [fsLocateExecutable("maxlev"), "gj1132_ribas.json", "--workers", "-1"],
        cwd=MAXLEV_DIR,
        check=True,
        env=dictEnv,
    )
    print("  MaxLEV completed.")


def fnGenerateFigure06(dictFigConfig, dictGlobalConfig):
    """Figure 6: sampler_comparison — emcee + dynesty posterior corner plot."""
    sMode = dictFigConfig["sMode"]
    sExtension = fsResolveOutputExtension(sMode)
    sOutputFile = fsResolveOutputPath(
        dictGlobalConfig["sOutputDirectory"], "sampler_comparison" + sExtension
    )

    if sMode in ("full", "standards"):
        fnEnsureMaxLevResults()
        sGenerated = os.path.join(
            POSTERIORS_DIR, "output", "sampler_comparison" + sExtension
        )
        fnRunSubprocess("gj1132_alabi.py", POSTERIORS_DIR, [sGenerated])
        fnCopyFigureFile(sGenerated, sOutputFile)
        return

    for sCandidate in [
        os.path.join(POSTERIORS_DIR, "output", "sampler_comparison.pdf"),
        os.path.join(POSTERIORS_DIR, "output", "sampler_comparison.png"),
    ]:
        if os.path.exists(sCandidate):
            fnCopyFigureFile(sCandidate, sOutputFile)
            return

    raise FileNotFoundError(
        "No sampler_comparison figure found. Set sMode to 'full' to generate."
    )


def fnGenerateFigure07(dictFigConfig, dictGlobalConfig):
    """Figure 7: EngleAgeHist.pdf — Engle age distribution."""
    sOutputFile = fsResolveOutputPath(
        dictGlobalConfig["sOutputDirectory"], "EngleAgeHist.pdf"
    )
    with fcontextWorkingDirectory(ENGLE_AGE_DIR):
        fnRunSubprocess("age.py", ENGLE_AGE_DIR)

    fnCopyFigureFile(
        os.path.join(ENGLE_AGE_DIR, "EngleAgeHist.pdf"), sOutputFile
    )
    plt.close("all")


def fnGenerateFigure08(dictFigConfig, dictGlobalConfig):
    """Figure 8: XUVEvol.pdf — 100 XUV evolution realizations."""
    sMode = dictFigConfig["sMode"]
    sOutputFile = fsResolveOutputPath(
        dictGlobalConfig["sOutputDirectory"], "XUVEvol.pdf"
    )
    sEngleOutput = os.path.join(
        CUMULATIVE_XUV_DIR, "EngleBarnes", "output"
    )
    sRibasOutput = os.path.join(
        CUMULATIVE_XUV_DIR, "RibasBarnes", "output"
    )

    bEngleExists = fbCheckVconvergeOutputDirectories(sEngleOutput)
    bRibasExists = fbCheckVconvergeOutputDirectories(sRibasOutput)

    bNeedsComputation = sMode in ("full", "standards")

    if (not bEngleExists or not bRibasExists) and not bNeedsComputation:
        raise FileNotFoundError(
            "XUVEvol requires vplanet output subdirectories "
            "(gj1132.star.forward files) in EngleBarnes/output/ and "
            "RibasBarnes/output/. Set sMode to 'full' or 'standards' to regenerate."
        )

    if bNeedsComputation:
        fnRunVconvergePipeline("EngleBarnes", dictGlobalConfig)
        fnRunVconvergePipeline("RibasBarnes", dictGlobalConfig)

    with fcontextWorkingDirectory(CUMULATIVE_XUV_DIR):
        sys.path.insert(0, CUMULATIVE_XUV_DIR)
        from plot_xuv_evolution import main as fnPlotXuvEvolution

        fnPlotXuvEvolution()

    fnCopyFigureFile(
        os.path.join(CUMULATIVE_XUV_DIR, "XUVEvol.pdf"), sOutputFile
    )
    plt.close("all")


def fnGenerateFigure09(dictFigConfig, dictGlobalConfig):
    """Figure 9: CumulativeXUV_Multi + CosmicShoreline."""
    sMode = dictFigConfig["sMode"]
    sExtension = fsResolveOutputExtension(sMode)
    sOutDir = dictGlobalConfig["sOutputDirectory"]

    # Left panel — cumulative XUV histogram
    sHistFile = os.path.join(
        CUMULATIVE_XUV_DIR, "GJ1132b_CumulativeXUV_Multi" + sExtension
    )
    fnRunSubprocess("vconverge_hist.py", CUMULATIVE_XUV_DIR, [sHistFile])
    fnCopyFigureFile(
        sHistFile,
        fsResolveOutputPath(sOutDir, "GJ1132b_CumulativeXUV_Multi" + sExtension),
    )

    # Right panel — cosmic shoreline
    sShorelineFile = os.path.join(
        COSMIC_SHORELINE_DIR, "CosmicShoreline" + sExtension
    )
    fnRunSubprocess("makeplot.py", COSMIC_SHORELINE_DIR, [sShorelineFile])
    fnCopyFigureFile(
        sShorelineFile,
        fsResolveOutputPath(sOutDir, "CosmicShoreline" + sExtension),
    )
    plt.close("all")


def fnGenerateFigure10(dictFigConfig, dictGlobalConfig):
    """Figure 10: ErrorSourceComparison — error decomposition."""
    sMode = dictFigConfig["sMode"]
    sExtension = fsResolveOutputExtension(sMode)
    sErrorFile = os.path.join(
        CUMULATIVE_XUV_DIR, "GJ1132b_ErrorSourceComparison" + sExtension
    )
    sOutputFile = fsResolveOutputPath(
        dictGlobalConfig["sOutputDirectory"],
        "GJ1132b_ErrorSourceComparison" + sExtension,
    )
    fnRunSubprocess(
        "error_source_compare.py", CUMULATIVE_XUV_DIR, [sErrorFile]
    )
    fnCopyFigureFile(sErrorFile, sOutputFile)
    plt.close("all")


def fnGenerateFigure11(dictFigConfig, dictGlobalConfig):
    """Figure 11: FitComparison.pdf — Kepler vs TESS comparison."""
    sOutputFile = fsResolveOutputPath(
        dictGlobalConfig["sOutputDirectory"], "FitComparison.pdf"
    )
    dictTess = fdictRunTessAnalysis(dictGlobalConfig)

    sys.path.insert(0, TESS_DIR)
    from flares_from_TESS import plot_alpha_beta_comparison

    dAlpha = dictTess["dictFitResults"]["alpha"]
    dBeta = dictTess["dictFitResults"]["beta"]

    with fcontextWorkingDirectory(TESS_DIR):
        plot_alpha_beta_comparison(dAlpha, dBeta, dictTess["dictFitResults"])
    fnCopyFigureFile(
        os.path.join(TESS_DIR, "FitComparison.pdf"), sOutputFile
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# Pipeline helpers (for 'full' mode)
# ---------------------------------------------------------------------------


def fbCheckVconvergeOutputDirectories(sOutputPath):
    """Return True if simulation subdirectories with forward files exist."""
    if not os.path.isdir(sOutputPath):
        return False
    for sEntry in os.listdir(sOutputPath):
        sSubdir = os.path.join(sOutputPath, sEntry)
        if os.path.isdir(sSubdir):
            sForwardFile = os.path.join(sSubdir, "gj1132.star.forward")
            if os.path.exists(sForwardFile):
                return True
    return False


def fnPreparePriorFiles(sModelDir):
    """Copy prior files into *sModelDir* if they are missing."""
    sFlareSrc = os.path.join(KEPLER_FLARES_DIR, "flare_mcmc_samples.npy")
    sFlareDst = os.path.join(sModelDir, "flares_variable_slope.npy")
    if not os.path.exists(sFlareDst) and os.path.exists(sFlareSrc):
        shutil.copy2(sFlareSrc, sFlareDst)
        print(f"  Copied flare priors -> {sFlareDst}")

    sAgeSrc = os.path.join(REPO_ROOT, "priors", "age_samples.txt")
    sAgeDst = os.path.join(sModelDir, "age_samples.txt")
    if not os.path.exists(sAgeDst) and os.path.exists(sAgeSrc):
        shutil.copy2(sAgeSrc, sAgeDst)
        print(f"  Copied age priors -> {sAgeDst}")

    sDynestySrc = os.path.join(REPO_ROOT, "priors", "dynesty_transform_final.npy")
    sDynestyDst = os.path.join(sModelDir, "dynesty_transform_final.npy")
    if not os.path.exists(sDynestyDst) and os.path.exists(sDynestySrc):
        shutil.copy2(sDynestySrc, sDynestyDst)
        print(f"  Copied dynesty priors -> {sDynestyDst}")


def fnBackupConvergedJson(sModelDir):
    """Back up Converged_Param_Dictionary.json before a full pipeline run."""
    sJsonPath = os.path.join(sModelDir, "output", "Converged_Param_Dictionary.json")
    sBackupPath = os.path.join(sModelDir, ".Converged_Param_Dictionary.json.bak")
    if os.path.exists(sJsonPath):
        shutil.copy2(sJsonPath, sBackupPath)
        print(f"  Backed up {sJsonPath}")
    return sBackupPath


def fnRestoreConvergedJson(sModelDir, sBackupPath):
    """Restore backed-up Converged_Param_Dictionary.json after pipeline failure."""
    sJsonPath = os.path.join(sModelDir, "output", "Converged_Param_Dictionary.json")
    if os.path.exists(sBackupPath):
        os.makedirs(os.path.dirname(sJsonPath), exist_ok=True)
        shutil.copy2(sBackupPath, sJsonPath)
        os.remove(sBackupPath)
        print(f"  Restored {sJsonPath} from backup")


def fnBuildNativeBinaryEnvironment():
    """Return environment dict with native vplanet binary on PATH."""
    dictEnv = os.environ.copy()
    dictEnv["PATH"] = VPLANET_NATIVE_BIN_DIR + ":" + dictEnv.get("PATH", "")
    return dictEnv


def fnRunVconvergePipeline(sModelName, dictGlobalConfig):
    """Run vspace/multiplanet/vconverge for *sModelName* subdirectory."""
    sModelDir = os.path.join(CUMULATIVE_XUV_DIR, sModelName)
    sVconvergeInput = os.path.join(sModelDir, "vconverge.in")

    if not os.path.exists(sVconvergeInput):
        raise FileNotFoundError(f"No vconverge.in in {sModelDir}")

    fnPreparePriorFiles(sModelDir)
    sBackupPath = fnBackupConvergedJson(sModelDir)

    print(f"  Running vconverge pipeline for {sModelName}...")
    sVconvergeExe = fsLocateExecutable("vconverge")
    dictEnv = fnBuildNativeBinaryEnvironment()
    try:
        subprocess.run(
            [sVconvergeExe, "vconverge.in"],
            cwd=sModelDir,
            check=True,
            env=dictEnv,
        )
        print(f"  Completed vconverge for {sModelName}")
        if os.path.exists(sBackupPath):
            os.remove(sBackupPath)
    except subprocess.CalledProcessError:
        fnRestoreConvergedJson(sModelDir, sBackupPath)
        raise


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

GENERATORS = {
    "figure01": fnGenerateFigure01,
    "figure02": fnGenerateFigure02,
    "figure03": fnGenerateFigure03,
    "figure04": fnGenerateFigure04,
    "figure05": fnGenerateFigure05,
    "figure06": fnGenerateFigure06,
    "figure07": fnGenerateFigure07,
    "figure08": fnGenerateFigure08,
    "figure09": fnGenerateFigure09,
    "figure10": fnGenerateFigure10,
    "figure11": fnGenerateFigure11,
}

# Execution order respecting dependencies
EXECUTION_ORDER = [
    "figure05",
    "figure07",
    "figure01",
    "figure02",
    "figure03",
    "figure04",
    "figure11",
    "figure09",
    "figure10",
    "figure08",
    "figure06",
]


def fnOrchestrate(dictConfig):
    """Generate all enabled figures in dependency order."""
    dictFigures = dictConfig["dictFigures"]
    listResults = []

    for sKey in EXECUTION_ORDER:
        dictFigure = dictFigures.get(sKey)
        if dictFigure is None or not dictFigure.get("bEnabled", True):
            continue

        sDescription = dictFigure.get("sDescription", sKey)
        print(f"\n{'=' * 60}")
        print(f"Generating {sKey}: {sDescription}")
        print(f"  Mode: {dictFigure['sMode']}")
        print(f"{'=' * 60}")

        try:
            GENERATORS[sKey](dictFigure, dictConfig)
            listResults.append((sKey, True, ""))
            print(f"  SUCCESS: {sKey}")
        except Exception as error:
            sErrorMessage = str(error)
            listResults.append((sKey, False, sErrorMessage))
            print(f"  FAILED: {sKey} — {sErrorMessage}")
            traceback.print_exc()

    fnPrintSummary(listResults)
    return listResults


def fnPrintSummary(listResults):
    """Print a table summarising the outcome of each figure."""
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    iPass = 0
    iFail = 0
    for sKey, bSuccess, sError in listResults:
        sStatus = "PASS" if bSuccess else "FAIL"
        if bSuccess:
            iPass += 1
        else:
            iFail += 1
        print(f"  {sKey:12s}  {sStatus:4s}  {sError[:50]}")
    print(f"\nTotal: {iPass} passed, {iFail} failed out of {len(listResults)}")


def fbVerifyAllFigures(dictConfig):
    """Return True when every expected file exists and is non-trivial."""
    sOutputDir = dictConfig["sOutputDirectory"]
    iMissing = 0
    for sKey, listBaseNames in EXPECTED_FILES.items():
        dictFigure = dictConfig["dictFigures"].get(sKey, {})
        if not dictFigure.get("bEnabled", True):
            continue
        sMode = dictFigure.get("sMode", "full")
        for sBaseName in listBaseNames:
            if sKey in FIXED_PDF_FIGURES:
                sExtension = ".pdf"
            else:
                sExtension = fsResolveOutputExtension(sMode)
            sPath = os.path.join(REPO_ROOT, sOutputDir, sBaseName + sExtension)
            if not os.path.exists(sPath):
                print(f"  MISSING: {sPath}")
                iMissing += 1
            elif os.path.getsize(sPath) < 1024:
                iBytes = os.path.getsize(sPath)
                print(f"  WARNING: {sPath} is only {iBytes} bytes")
    return iMissing == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Parse arguments and run the orchestration pipeline."""
    os.environ["PATH"] = VPLANET_NATIVE_BIN_DIR + ":" + os.environ.get("PATH", "")

    parser = argparse.ArgumentParser(
        description="Generate all figures for the GJ 1132 XUV paper."
    )
    parser.add_argument(
        "--config",
        default="constrain.json",
        help="Path to the constrain.json configuration file.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that all expected figure files exist.",
    )
    args = parser.parse_args()

    sConfigPath = os.path.join(REPO_ROOT, args.config)
    dictConfig = fdictLoadConfiguration(sConfigPath)

    sOutputDir = os.path.join(REPO_ROOT, dictConfig["sOutputDirectory"])
    os.makedirs(sOutputDir, exist_ok=True)

    if args.verify_only:
        bAllPresent = fbVerifyAllFigures(dictConfig)
        sys.exit(0 if bAllPresent else 1)

    listResults = fnOrchestrate(dictConfig)
    print("\nVerifying output files...")
    fbVerifyAllFigures(dictConfig)

    bAllPassed = all(bSuccess for _, bSuccess, _ in listResults)
    sys.exit(0 if bAllPassed else 1)


if __name__ == "__main__":
    main()
