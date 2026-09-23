#!/bin/bash
#SBATCH --job-name=cardiology_survival
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=12:00:00
#SBATCH --tmp=200gb
##SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kuana001@umn.edu
#SBATCH --nodes=1
##SBATCH --partition=msibigmem
#SBATCH --partition=ag2tb
##SBATCH --partition=msilong
##SBATCH --partition=interactive-long
#SBATCH --error=survival_summary_%j.err
#SBATCH --output=survival_summary_%j.out

set -euo pipefail

# Use the directory from which this job was submitted. This avoids hard-coding
# the project folder name when the directory is renamed.
PROJECT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
PYTHON_SCRIPT="cardiology_hf_survival_analysis.py"
INPUT_XLSX="synthetic_cardiology_hfpef_cohort.xlsx"
HORIZON_DAYS=365
TEST_SIZE=0.25
COX_PENALIZER=0.10

echo "Job ID: ${SLURM_JOB_ID:-not_available}"
echo "Working directory: ${PROJECT_DIR}"
echo "Input workbook: ${INPUT_XLSX}"
echo "Follow-up horizon: ${HORIZON_DAYS} days"
cd "${PROJECT_DIR}"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script not found: ${PROJECT_DIR}/${PYTHON_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${INPUT_XLSX}" ]]; then
    echo "ERROR: Excel dataset not found: ${PROJECT_DIR}/${INPUT_XLSX}" >&2
    exit 1
fi

# Fail early with a clear message if the survival-analysis packages are absent.
python -c "import lifelines, pandas, sklearn, openpyxl, matplotlib, joblib" || {
    echo "ERROR: Missing Python dependencies. Install lifelines, pandas, numpy," >&2
    echo "       scikit-learn, openpyxl, matplotlib, and joblib." >&2
    exit 1
}

echo "Starting time-to-first HF hospitalization analysis"
python "${PYTHON_SCRIPT}" \
    --input-xlsx "${INPUT_XLSX}" \
    --endpoint hf_hospitalization \
    --horizon-days "${HORIZON_DAYS}" \
    --test-size "${TEST_SIZE}" \
    --penalizer "${COX_PENALIZER}" \
    --output-dir survival_hf_results

echo "Starting time-to-death analysis"
python "${PYTHON_SCRIPT}" \
    --input-xlsx "${INPUT_XLSX}" \
    --endpoint death \
    --horizon-days "${HORIZON_DAYS}" \
    --test-size "${TEST_SIZE}" \
    --penalizer "${COX_PENALIZER}" \
    --output-dir survival_death_results

echo "Starting time-to-first hospitalization-or-death analysis"
python "${PYTHON_SCRIPT}" \
    --input-xlsx "${INPUT_XLSX}" \
    --endpoint composite \
    --horizon-days "${HORIZON_DAYS}" \
    --test-size "${TEST_SIZE}" \
    --penalizer "${COX_PENALIZER}" \
    --output-dir survival_composite_results

echo "All three survival analyses completed successfully"
