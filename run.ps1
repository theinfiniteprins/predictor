<#
  One command for the whole daily flow. Run from anywhere:

      D:\Predictor\run.ps1

  Options:
      -Tune 200        run an Optuna hyperparameter search (slow; do occasionally)
      -K 0.6           override the barrier multiplier for this run
      -Holdout 14      train only through (last day - 14) so the tail is out-of-sample
      -SkipPull        don't git pull
      -SkipBackfill    don't re-hit yfinance (use existing raw data)

  What it does, in order:
      git pull  ->  refresh raw data  ->  build dataset  ->  paper-log yesterday's
      calls (before retraining)  ->  retrain  ->  backtest  ->  print the report
#>
param(
    [int]$Tune = 0,
    [double]$K = 0,
    [int]$Holdout = 0,
    [switch]$SkipPull,
    [switch]$SkipBackfill
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "venv missing. First-time setup:" -ForegroundColor Red
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -e ."
    exit 1
}

function Step($name, [scriptblock]$block) {
    Write-Host "`n========== $name ==========" -ForegroundColor Cyan
    & $block
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        Write-Host "  ($name exited $LASTEXITCODE - continuing)" -ForegroundColor Yellow
    }
}

$started = Get-Date

if (-not $SkipPull)     { Step "git pull (collector data)" { git pull --rebase --autostash } }
if (-not $SkipBackfill) { Step "refresh raw data"          { & $py scripts\run_backfill.py --all } }

$kArg = @(); if ($K -gt 0) { $kArg = @("--k", $K) }
Step "build dataset (consolidate + label + features)" { & $py scripts\build_dataset.py @kArg }

# log yesterday's resolved calls against the CURRENT model, before it gets retrained
Step "paper-trade log" { & $py scripts\paper_log.py --no-build }

$trainArgs = @()
if ($Tune -gt 0)    { $trainArgs += @("--tune", $Tune) }
if ($Holdout -gt 0) { $trainArgs += @("--holdout-days", $Holdout) }
Step "retrain (walk-forward primary + meta)" { & $py scripts\train.py @trainArgs }

Step "backtest" { & $py scripts\backtest.py }

Write-Host "`n"
& $py scripts\report.py

$mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
Write-Host "`ndone in $mins min.  full logs: $root\logs\predictor.log" -ForegroundColor DarkGray
