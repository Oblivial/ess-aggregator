# Dev environment setup for developing py-ess and ess-aggregator together.
#
# Creates (or reuses) a single virtual environment at the workspace root and
# installs both sibling packages in *editable* mode, so edits to either
# repo's source take effect immediately without reinstalling. This avoids
# the classic "stale non-editable install silently shadows the local
# checkout" trap.
#
# Usage (from anywhere, e.g. from the ess-aggregator checkout):
#   .\scripts\dev-setup.ps1

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$aggregatorDir = Split-Path -Parent $scriptDir
$workspaceRoot = Split-Path -Parent $aggregatorDir
$pyEssDir = Join-Path $workspaceRoot "py-ess"
$venvDir = Join-Path $workspaceRoot ".venv"

foreach ($dir in @($pyEssDir, $aggregatorDir)) {
    if (-not (Test-Path $dir)) {
        throw "Expected sibling project directory not found: $dir"
    }
}

if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment at $venvDir"
    python -m venv $venvDir
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"

Write-Host "Installing py-ess (editable, with dev extras) from $pyEssDir"
& $venvPython -m pip install -e "$pyEssDir[dev]"

Write-Host "Installing ess-aggregator (editable, with dev extras) from $aggregatorDir, without re-pulling py-ess from git"
& $venvPython -m pip install -e "$aggregatorDir" --no-deps
& $venvPython -m pip install pandas numpy pytest pytest-cov

Write-Host ""
Write-Host "Done. Activate the environment with:"
Write-Host "  $venvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Verify both packages resolve to their local checkouts with:"
Write-Host "  pip show py-ess ess-aggregator"
Write-Host "(look for 'Editable project location' pointing at your checkouts)"
