<#
.SYNOPSIS
    Convenient task runner script for Windows PowerShell.
.DESCRIPTION
    Mirrors Makefile commands for Windows developers.
.EXAMPLE
    .\run.ps1 test
    .\run.ps1 train
    .\run.ps1 app
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("help", "install", "test", "train", "train-sample", "app", "lint", "clean")]
    [string]$Command = "help"
)

switch ($Command) {
    "help" {
        Write-Host "Usage: .\run.ps1 <command>" -ForegroundColor Cyan
        Write-Host "Commands:"
        Write-Host "  test         - Run pytest test suite"
        Write-Host "  train        - Train XGBoost on full dataset"
        Write-Host "  train-sample - Train XGBoost on 5% sample"
        Write-Host "  app          - Launch Streamlit dashboard"
        Write-Host "  lint         - Check code quality with ruff"
        Write-Host "  clean        - Remove __pycache__ and temp artifacts"
    }
    "test" {
        python -m pytest tests/ -v
    }
    "train" {
        python train_model.py --model xgboost
    }
    "train-sample" {
        python train_model.py --model xgboost --sample 0.05
    }
    "app" {
        streamlit run app/streamlit_app.py
    }
    "lint" {
        python -m ruff check src tests app
    }
    "clean" {
        Get-ChildItem -Path . -Recurse -Include "__pycache__", ".pytest_cache" | Remove-Item -Recurse -Force
        Write-Host "Cleaned build artifacts." -ForegroundColor Green
    }
}