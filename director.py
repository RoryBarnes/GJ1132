#!/usr/bin/env python3
"""
Generic command-driven pipeline for generating paper figures.

Reads script.json, which defines a sequence of scenes. Each scene has
a working directory, setup commands (heavy computation), always-run
commands (plotting), and expected output files. Output files from
earlier scenes are available as {SceneNN.stem} variables in later
scenes.

Usage:
    python director.py
    python director.py --config script.json
    python director.py --verify-only
"""

import argparse
import json
import multiprocessing
import os
import re
import subprocess
import sys
import threading


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Variable interpolation
# ---------------------------------------------------------------------------


def fsResolveVariables(sTemplate, dictVariables):
    """Replace {name} tokens in *sTemplate* with values from *dictVariables*."""

    def fnReplace(match):
        sToken = match.group(1)
        if sToken in dictVariables:
            return str(dictVariables[sToken])
        raise KeyError(f"Unresolved variable: {{{sToken}}}")

    return re.sub(r"\{([^}]+)\}", fnReplace, sTemplate)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def fdictLoadScript(sScriptPath):
    """Load and validate *sScriptPath*, returning the parsed dictionary."""
    if not os.path.isfile(sScriptPath):
        print(f"ERROR: Configuration file not found: {sScriptPath}")
        sys.exit(1)
    with open(sScriptPath, "r") as fileHandle:
        dictScript = json.load(fileHandle)
    if not fbValidateScript(dictScript):
        print("ERROR: Invalid script.json.")
        sys.exit(1)
    return dictScript


def fbValidateScript(dictScript):
    """Return True when all required keys and scene structures are valid."""
    for sKey in ("sPlotDirectory", "listScenes"):
        if sKey not in dictScript:
            print(f"Missing required key: {sKey}")
            return False
    for iIndex, dictScene in enumerate(dictScript["listScenes"]):
        sLabel = f"Scene{iIndex + 1:02d}"
        for sField in ("sName", "sDirectory", "saCommands", "saOutputFiles"):
            if sField not in dictScene:
                print(f"{sLabel}: missing required field '{sField}'")
                return False
    return True


# ---------------------------------------------------------------------------
# Global variable construction
# ---------------------------------------------------------------------------


def fiResolveCoreCount(iRequested):
    """Return a usable core count from the requested value (-1 = auto)."""
    iTotal = multiprocessing.cpu_count()
    if iRequested == -1:
        return max(1, iTotal - 1)
    return min(iRequested, iTotal)


def fdictBuildGlobalVariables(dictScript):
    """Extract top-level script.json keys into a variables dictionary."""
    sPlotDirectory = os.path.join(REPO_ROOT, dictScript["sPlotDirectory"])
    os.makedirs(sPlotDirectory, exist_ok=True)
    return {
        "sPlotDirectory": sPlotDirectory,
        "sRepoRoot": REPO_ROOT,
        "iNumberOfCores": fiResolveCoreCount(
            dictScript.get("iNumberOfCores", -1)),
        "sFigureType": dictScript.get("sFigureType", "pdf").lower(),
    }


# ---------------------------------------------------------------------------
# Executable name extraction
# ---------------------------------------------------------------------------


def fsExtractExecutableName(sCommand):
    """Extract a display name for the executable from a command string."""
    listTokens = sCommand.split()
    if not listTokens:
        return "unknown"
    sFirst = os.path.basename(listTokens[0])
    if sFirst == "python" and len(listTokens) > 1:
        return os.path.basename(listTokens[1])
    if "&&" in listTokens:
        iIndex = listTokens.index("&&")
        if iIndex + 1 < len(listTokens):
            return os.path.basename(listTokens[iIndex + 1])
    return sFirst


# ---------------------------------------------------------------------------
# Command execution with prefixed logging
# ---------------------------------------------------------------------------


def fnStreamPrefixedOutput(stream, sPrefix):
    """Read *stream* line-by-line and print each line with *sPrefix*."""
    for sLine in stream:
        print(f"{sPrefix} {sLine}", end="", flush=True)
    stream.close()


def fnExecuteCommand(sCommand, sWorkingDirectory, sSceneName):
    """Run *sCommand* via shell, streaming prefixed output."""
    if not os.path.isdir(sWorkingDirectory):
        raise FileNotFoundError(
            f"Working directory does not exist: {sWorkingDirectory}")
    sExecutable = fsExtractExecutableName(sCommand)
    sPrefix = f"[{sSceneName}][{sExecutable}]"
    print(f"  Running: {sCommand}")

    process = subprocess.Popen(
        sCommand, shell=True, cwd=sWorkingDirectory,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=os.environ.copy(),
    )
    threadOut = threading.Thread(
        target=fnStreamPrefixedOutput, args=(process.stdout, sPrefix))
    threadErr = threading.Thread(
        target=fnStreamPrefixedOutput, args=(process.stderr, sPrefix))
    threadOut.start()
    threadErr.start()
    threadOut.join()
    threadErr.join()
    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"Exit code {process.returncode}: {sCommand}")


# ---------------------------------------------------------------------------
# Scene execution
# ---------------------------------------------------------------------------


def fnExecuteScene(dictScene, dictVariables):
    """Execute all commands in a scene, respecting bPlotOnly."""
    sDirectory = fsResolveVariables(
        dictScene["sDirectory"], dictVariables)
    sAbsDirectory = os.path.join(REPO_ROOT, sDirectory)
    bPlotOnly = dictScene.get("bPlotOnly", True)

    if not bPlotOnly:
        for sCommand in dictScene.get("saSetupCommands", []):
            sResolved = fsResolveVariables(sCommand, dictVariables)
            fnExecuteCommand(sResolved, sAbsDirectory, dictScene["sName"])

    for sCommand in dictScene["saCommands"]:
        sResolved = fsResolveVariables(sCommand, dictVariables)
        fnExecuteCommand(sResolved, sAbsDirectory, dictScene["sName"])


def fsResolveOutputPath(sOutputFile, dictVariables, sAbsDirectory):
    """Resolve an output file spec to an absolute path."""
    sResolvedPath = fsResolveVariables(sOutputFile, dictVariables)
    if os.path.isabs(sResolvedPath):
        return sResolvedPath
    return os.path.join(sAbsDirectory, sResolvedPath)


def fnRegisterSceneOutputs(dictScene, dictVariables, sSceneLabel):
    """Verify output files exist and register them as variables."""
    sDirectory = fsResolveVariables(
        dictScene["sDirectory"], dictVariables)
    sAbsDirectory = os.path.join(REPO_ROOT, sDirectory)

    for sOutputFile in dictScene["saOutputFiles"]:
        sAbsPath = fsResolveOutputPath(
            sOutputFile, dictVariables, sAbsDirectory)
        if not os.path.exists(sAbsPath):
            raise FileNotFoundError(
                f"{sSceneLabel} expected output not found: {sAbsPath}")
        iFileSize = os.path.getsize(sAbsPath)
        if iFileSize < 1024:
            print(f"  WARNING: {sAbsPath} is only {iFileSize} bytes")
        sStem = os.path.splitext(os.path.basename(sAbsPath))[0]
        sKey = f"{sSceneLabel}.{sStem}"
        dictVariables[sKey] = sAbsPath
        print(f"  Registered: {{{sKey}}} -> {sAbsPath}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def fnPrintSceneBanner(sSceneLabel, dictScene, dictVariables):
    """Print a visual separator with scene metadata."""
    print(f"\n{'=' * 60}")
    print(f"{sSceneLabel}: {dictScene['sName']}")
    print(f"  bPlotOnly: {dictScene.get('bPlotOnly', True)}"
          f" | sFigureType: {dictVariables['sFigureType']}"
          f" | sDirectory: {dictScene['sDirectory']}")
    print(f"{'=' * 60}")


def fnPrintSummary(listResults):
    """Print a table summarising the outcome of each scene."""
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    iPass = sum(1 for _, _, bOk, _ in listResults if bOk)
    iFail = len(listResults) - iPass
    for sLabel, sName, bSuccess, sError in listResults:
        sStatus = "PASS" if bSuccess else "FAIL"
        sSuffix = f"  {sError[:50]}" if sError else ""
        print(f"  {sLabel} {sName:40s}  {sStatus:4s}{sSuffix}")
    print(f"\nTotal: {iPass} passed, {iFail} failed out of {len(listResults)}")


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------


def fnRunVerifyOnly(dictScript, dictVariables):
    """Check that all expected output files exist without executing."""
    listResults = []
    for iScene, dictScene in enumerate(dictScript["listScenes"]):
        if not dictScene.get("bEnabled", True):
            continue
        sLabel = f"Scene{iScene + 1:02d}"
        try:
            fnRegisterSceneOutputs(dictScene, dictVariables, sLabel)
            listResults.append((sLabel, dictScene["sName"], True, ""))
        except FileNotFoundError as error:
            listResults.append(
                (sLabel, dictScene["sName"], False, str(error)))
    fnPrintSummary(listResults)
    return all(bOk for _, _, bOk, _ in listResults)


def fnRunPipeline(dictScript, dictVariables):
    """Execute all enabled scenes, halting on first failure."""
    listResults = []
    for iScene, dictScene in enumerate(dictScript["listScenes"]):
        if not dictScene.get("bEnabled", True):
            continue
        sLabel = f"Scene{iScene + 1:02d}"
        dictVariables["sFigureType"] = dictScene.get(
            "sFigureType", dictScript.get("sFigureType", "pdf")).lower()
        fnPrintSceneBanner(sLabel, dictScene, dictVariables)
        try:
            fnExecuteScene(dictScene, dictVariables)
            fnRegisterSceneOutputs(dictScene, dictVariables, sLabel)
            listResults.append((sLabel, dictScene["sName"], True, ""))
            print(f"  SUCCESS: {sLabel}")
        except Exception as error:
            listResults.append(
                (sLabel, dictScene["sName"], False, str(error)))
            print(f"  FAILED: {sLabel} — {error}")
            fnPrintSummary(listResults)
            return False
    fnPrintSummary(listResults)
    return all(bOk for _, _, bOk, _ in listResults)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def fnConfigureEnvironment(dictScript):
    """Set PATH and LIGHTKURVE_CACHE_DIR from script.json."""
    sVplanetDir = dictScript.get("sVplanetBinaryDirectory", "")
    if sVplanetDir:
        os.environ["PATH"] = sVplanetDir + ":" + os.environ.get("PATH", "")
    sTessCache = dictScript.get("sTessDataCache", "data/tess_cache")
    os.environ["LIGHTKURVE_CACHE_DIR"] = os.path.abspath(
        os.path.join(REPO_ROOT, sTessCache))


def main():
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate all figures from a script.json pipeline.")
    parser.add_argument(
        "--config", default="script.json",
        help="Path to the script.json configuration file.")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify that all expected output files exist.")
    args = parser.parse_args()

    sScriptPath = os.path.join(REPO_ROOT, args.config)
    dictScript = fdictLoadScript(sScriptPath)
    fnConfigureEnvironment(dictScript)
    dictVariables = fdictBuildGlobalVariables(dictScript)

    if args.verify_only:
        bSuccess = fnRunVerifyOnly(dictScript, dictVariables)
    else:
        bSuccess = fnRunPipeline(dictScript, dictVariables)
    sys.exit(0 if bSuccess else 1)


if __name__ == "__main__":
    main()
