#!/usr/bin/env python3
"""Time-to-event analysis for the index-aligned HFpEF cohort.

Endpoints
---------
hf_hospitalization
    First inpatient hospitalization with a primary HF diagnosis. Death before
    hospitalization is a competing event. A cause-specific Cox model is fit,
    and Aalen-Johansen cumulative incidence curves are reported.
death
    All-cause death.
composite
    First occurrence of HF hospitalization or death.

All predictors come from on/before the HFpEF index date. Follow-up is capped at
365 days by default. Patients without an observed event are right-censored at
their last known follow-up; they are not automatically labeled non-events.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
except ImportError as exc:
    raise ImportError(
        "lifelines is required. Install dependencies with: pip install "
        "lifelines pandas numpy scikit-learn openpyxl matplotlib joblib"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "synthetic_cardiology_hfpef_cohort.xlsx"

# Deliberate baseline allow-list. No post-index outcome or censoring field can
# enter the model. Raw ECG waveforms and echo videos are not used.
BASELINE_FEATURES = [
    "AGE_AT_INDEX", "SEX", "ETHNICITY", "RACE",
    "BASELINE_BMI", "BMI_SLOPE_365D", "BMI_VARIABILITY_365D",
    "BLOOD_PRESSURE_SYSTOLIC", "BLOOD_PRESSURE_DIASTOLIC", "PULSE",
    "PULSE_OXYMETRY", "RESPIRATORY_RATE", "SBP_MEAN_90D", "SBP_MAX_90D",
    "SBP_SLOPE_90D", "VITAL_MEASUREMENT_COUNT_90D",
    "BASELINE_PROBNP_RESULT_VALUE_NUM", "PROBNP_MEAN_90D", "PROBNP_MAX_90D",
    "PROBNP_SLOPE_90D", "BASELINE_EGFR", "EGFR_MEAN_90D", "EGFR_MIN_90D",
    "EGFR_SLOPE_90D", "BASELINE_CREATININE_MG_DL", "BASELINE_BUN_MG_DL",
    "BASELINE_SODIUM_MMOL_L", "BASELINE_POTASSIUM_MMOL_L",
    "BASELINE_HEMOGLOBIN_G_DL", "BASELINE_ALBUMIN_G_DL", "BASELINE_HBA1C",
    "BASELINE_TROPONIN_NG_L",
    "BASELINE_LVEF", "BASELINE_E_OVER_E_PRIME", "BASELINE_TR_VELOCITY",
    "BASELINE_LAVI", "BASELINE_GLS", "BASELINE_LV_MASS_INDEX",
    "BASELINE_DIASTOLIC_FUNCTION_GRADE", "BASELINE_PASP_MMHG",
    "BASELINE_TAPSE_CM", "BASELINE_RV_S_PRIME_CM_S", "BASELINE_RV_FAC_PERCENT",
    "BASELINE_RV_FUNCTION_GRADE", "BASELINE_MITRAL_REGURGITATION_GRADE",
    "BASELINE_TRICUSPID_REGURGITATION_GRADE",
    "ECG_HEART_RATE_BPM", "ECG_PR_INTERVAL_MS", "ECG_QRS_DURATION_MS",
    "ECG_QTC_INTERVAL_MS", "ECG_ATRIAL_FIBRILLATION_FLAG", "ECG_LBBB_FLAG",
    "ECG_RBBB_FLAG", "ECG_AV_BLOCK_FLAG", "ECG_LVH_FLAG",
    "ECG_PACED_RHYTHM_FLAG", "ECG_ABNORMAL_FLAG",
    "PRIOR_HF_HOSPITALIZATION_COUNT_365D",
    "PRIOR_ALL_CAUSE_HOSPITALIZATION_COUNT_365D", "PRIOR_ED_VISIT_COUNT_365D",
    "PRIOR_CARDIOLOGY_VISIT_COUNT_365D", "DAYS_SINCE_LAST_HF_HOSPITALIZATION",
    "PRIOR_TOTAL_INPATIENT_DAYS_365D", "PRIOR_ICU_ADMISSION_FLAG",
    "ATRIAL_FIBRILLATION_FLAG", "HYPERTENSION_FLAG", "TYPE2_DIABETES_FLAG",
    "CHRONIC_KIDNEY_DISEASE_FLAG", "CKD_STAGE", "CORONARY_ARTERY_DISEASE_FLAG",
    "PRIOR_MI_FLAG", "PRIOR_STROKE_FLAG", "COPD_FLAG", "SLEEP_APNEA_FLAG",
    "PULMONARY_HYPERTENSION_FLAG", "ANEMIA_FLAG", "CHARLSON_COMORBIDITY_INDEX",
    "ELIXHAUSER_SCORE", "HAS_SGLT2_ACTIVE_AT_INDEX",
    "HAS_INCRETIN_ACTIVE_AT_INDEX", "MEDICATION_EXPOSURE_GROUP",
    "LOOP_DIURETIC_ACTIVE_AT_INDEX", "MRA_ACTIVE_AT_INDEX",
    "ACEI_ARB_ARNI_ACTIVE_AT_INDEX", "BETA_BLOCKER_ACTIVE_AT_INDEX",
    "ANTICOAGULANT_ACTIVE_AT_INDEX", "STATIN_ACTIVE_AT_INDEX",
    "TOTAL_ACTIVE_MEDICATION_COUNT_AT_INDEX",
]


def resolve_input_workbook(path: Path) -> Path:
    """Resolve only the explicitly requested HFpEF cohort workbook."""
    requested = path.expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(SCRIPT_DIR / requested)
    for candidate in dict.fromkeys(item.resolve() for item in candidates):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Required input workbook not found. This program does not substitute a "
        "different Excel file. Checked the requested path and, for a relative "
        f"path, the script directory: {requested}. Supply the full path with "
        "--input-xlsx if the workbook is stored elsewhere."
    )


def load_workbook(path: Path, sheet_name: str) -> pd.DataFrame:
    resolved = resolve_input_workbook(path)
    print(f"Loading workbook: {resolved}")
    excel = pd.ExcelFile(resolved)
    if sheet_name not in excel.sheet_names:
        if sheet_name != "Synthetic SQL Output":
            raise ValueError(f"Sheet {sheet_name!r} not found: {excel.sheet_names}")
        warnings.warn(f"Using first sheet {excel.sheet_names[0]!r}.", stacklevel=2)
        sheet_name = excel.sheet_names[0]
    df = pd.read_excel(excel, sheet_name=sheet_name)
    df.columns = [str(column).strip().upper() for column in df.columns]
    if df.empty:
        raise ValueError("The selected sheet contains no rows.")
    return df


def add_age_at_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "AGE_AT_INDEX" not in result and {"BIRTH_DATE", "FIRST_RECORDED_HFPEF_INDEX_DATE"} <= set(result):
        birth = pd.to_datetime(result["BIRTH_DATE"], errors="coerce")
        index = pd.to_datetime(result["FIRST_RECORDED_HFPEF_INDEX_DATE"], errors="coerce")
        result["AGE_AT_INDEX"] = ((index - birth).dt.days / 365.25).round(1)
    return result


def numeric_column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def observation_days(df: pd.DataFrame, horizon: int) -> pd.Series:
    observed = numeric_column(df, "OUTCOME_OBSERVATION_DAYS")
    if {"CENSORING_DATETIME", "FIRST_RECORDED_HFPEF_INDEX_DATE"} <= set(df):
        censor = pd.to_datetime(df["CENSORING_DATETIME"], errors="coerce")
        index = pd.to_datetime(df["FIRST_RECORDED_HFPEF_INDEX_DATE"], errors="coerce")
        observed = observed.fillna((censor - index).dt.days)
    complete = numeric_column(df, "COMPLETE_365D_FOLLOWUP_FLAG").eq(1)
    observed = observed.mask(observed.isna() & complete, horizon)
    # If no follow-up field is available, the binary outcomes alone imply that
    # observed non-events were followed through the requested horizon.
    known_binary = pd.concat([
        numeric_column(df, "HF_HOSPITALIZATION_WITHIN_365D"),
        numeric_column(df, "DEATH_WITHIN_365D"),
    ], axis=1).notna().all(axis=1)
    observed = observed.mask(observed.isna() & known_binary, horizon)
    if observed.isna().any():
        raise ValueError(
            f"{int(observed.isna().sum())} rows lack usable follow-up duration. "
            "Provide OUTCOME_OBSERVATION_DAYS or censoring/index dates."
        )
    return observed.clip(lower=1, upper=horizon).astype(float)


def build_survival_outcome(df: pd.DataFrame, endpoint: str, horizon: int) -> pd.DataFrame:
    """Return duration and event codes; 0=censored, 1=event, 2=competing death."""
    hosp = numeric_column(df, "DAYS_INDEX_TO_FIRST_HF_HOSPITALIZATION")
    death = numeric_column(df, "DAYS_INDEX_TO_DEATH")
    followup = observation_days(df, horizon)
    valid_hosp = hosp.between(1, horizon, inclusive="both") & hosp.le(followup)
    valid_death = death.between(1, horizon, inclusive="both") & death.le(followup)

    duration = followup.copy()
    code = pd.Series(0, index=df.index, dtype=int)
    label = pd.Series("Censored", index=df.index, dtype="string")

    if endpoint == "hf_hospitalization":
        hf_first = valid_hosp & (~valid_death | hosp.le(death))
        death_first = valid_death & (~valid_hosp | death.lt(hosp))
        duration.loc[hf_first] = hosp.loc[hf_first]
        code.loc[hf_first] = 1
        label.loc[hf_first] = "HF hospitalization"
        duration.loc[death_first] = death.loc[death_first]
        code.loc[death_first] = 2
        label.loc[death_first] = "Death before HF hospitalization"
    elif endpoint == "death":
        duration.loc[valid_death] = death.loc[valid_death]
        code.loc[valid_death] = 1
        label.loc[valid_death] = "Death"
    elif endpoint == "composite":
        first_day = pd.concat([hosp.where(valid_hosp), death.where(valid_death)], axis=1).min(axis=1)
        event = first_day.notna()
        duration.loc[event] = first_day.loc[event]
        code.loc[event] = 1
        label.loc[event] = "HF hospitalization or death"
    else:
        raise ValueError(f"Unsupported endpoint: {endpoint}")

    return pd.DataFrame({
        "DURATION_DAYS": duration.clip(lower=1, upper=horizon),
        "EVENT_CODE": code,
        "EVENT": code.eq(1).astype(int),
        "EVENT_LABEL": label,
    }, index=df.index)


def column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [
        column for column in X
        if is_object_dtype(X[column].dtype) or is_string_dtype(X[column].dtype)
        or isinstance(X[column].dtype, pd.CategoricalDtype)
    ]
    return categorical, [column for column in X if column not in categorical]


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def safe_feature_names(names: np.ndarray) -> list[str]:
    used: dict[str, int] = {}
    output = []
    for raw in names:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw)).strip("_") or "feature"
        used[base] = used.get(base, 0) + 1
        output.append(base if used[base] == 1 else f"{base}_{used[base]}")
    return output


def save_curves(outcome: pd.DataFrame, endpoint: str, horizon: int, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    km = KaplanMeierFitter(label="Event-free survival")
    km.fit(outcome["DURATION_DAYS"], event_observed=outcome["EVENT"])
    km.plot_survival_function(ax=axes[0], ci_show=True)
    axes[0].set(title="Kaplan–Meier event-free survival", xlabel="Days after index", ylabel="Probability")
    axes[0].set_xlim(0, horizon)

    if endpoint == "hf_hospitalization":
        aj_hf = AalenJohansenFitter(seed=42)
        aj_hf.fit(outcome["DURATION_DAYS"], outcome["EVENT_CODE"], event_of_interest=1)
        aj_hf.plot_cumulative_density(ax=axes[1], label="HF hospitalization")
        if outcome["EVENT_CODE"].eq(2).any():
            aj_death = AalenJohansenFitter(seed=42)
            aj_death.fit(outcome["DURATION_DAYS"], outcome["EVENT_CODE"], event_of_interest=2)
            aj_death.plot_cumulative_density(ax=axes[1], label="Competing death")
        axes[1].set_title("Aalen–Johansen cumulative incidence")
        axes[1].set_ylabel("Cumulative incidence")
    else:
        cumulative = 1.0 - km.survival_function_
        axes[1].step(cumulative.index, cumulative.iloc[:, 0], where="post")
        axes[1].set(title="Cumulative event probability", ylabel="1 − survival")
    axes[1].set(xlabel="Days after index", xlim=(0, horizon))
    fig.suptitle(f"Survival analysis: {endpoint.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(output_dir / "survival_curves.png", dpi=180)
    plt.close(fig)


def run_analysis(
    df: pd.DataFrame, endpoint: str, horizon: int, output_dir: Path,
    test_size: float, penalizer: float, seed: int,
) -> dict:
    outcome = build_survival_outcome(df, endpoint, horizon)
    available = [c for c in BASELINE_FEATURES if c in df and not df[c].isna().all()]
    if not available:
        raise ValueError("No expected baseline predictors were found.")
    if outcome["EVENT"].sum() < 10:
        raise ValueError("At least 10 primary events are required for this analysis.")

    X = df[available].copy()
    ids = df.get("PATIENT_ID", pd.Series(df.index.astype(str), index=df.index)).astype(str)
    stratify = outcome["EVENT"] if outcome["EVENT"].value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=seed, stratify=stratify
    )
    categorical, numeric = column_types(X)
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder()),
        ]), categorical))
    preprocess = ColumnTransformer(transforms, remainder="drop", verbose_feature_names_out=True)
    train_matrix = preprocess.fit_transform(X.loc[train_idx])
    test_matrix = preprocess.transform(X.loc[test_idx])
    names = safe_feature_names(preprocess.get_feature_names_out())
    # Constant encoded columns add no information and can destabilize Cox fitting.
    keep = np.nanstd(train_matrix, axis=0) > 1e-12
    if not np.any(keep):
        raise ValueError("All preprocessed predictors are constant in the training set.")
    train_matrix = train_matrix[:, keep]
    test_matrix = test_matrix[:, keep]
    names = [name for name, selected in zip(names, keep) if selected]
    train_model = pd.DataFrame(train_matrix, columns=names, index=train_idx)
    test_model = pd.DataFrame(test_matrix, columns=names, index=test_idx)
    train_model[["DURATION_DAYS", "EVENT"]] = outcome.loc[train_idx, ["DURATION_DAYS", "EVENT"]]

    cox = CoxPHFitter(penalizer=penalizer)
    cox.fit(train_model, duration_col="DURATION_DAYS", event_col="EVENT", show_progress=False)
    risk_train = cox.predict_partial_hazard(train_model[names]).to_numpy().ravel()
    risk_test = cox.predict_partial_hazard(test_model[names]).to_numpy().ravel()
    c_index = concordance_index(
        outcome.loc[test_idx, "DURATION_DAYS"], -risk_test, outcome.loc[test_idx, "EVENT"]
    )
    q1, q2 = np.quantile(risk_train, [1 / 3, 2 / 3])
    risk_group = np.select([risk_test < q1, risk_test < q2], ["Low", "Intermediate"], default="High")

    output_dir.mkdir(parents=True, exist_ok=True)
    patient_outcome = pd.concat([ids.rename("PATIENT_ID"), outcome], axis=1)
    patient_outcome["DATA_SPLIT"] = np.where(patient_outcome.index.isin(test_idx), "test", "train")
    patient_outcome.to_csv(output_dir / "survival_patient_data.csv", index=False)
    predictions = pd.DataFrame({
        "PATIENT_ID": ids.loc[test_idx],
        "DURATION_DAYS": outcome.loc[test_idx, "DURATION_DAYS"],
        "EVENT_CODE": outcome.loc[test_idx, "EVENT_CODE"],
        "EVENT_LABEL": outcome.loc[test_idx, "EVENT_LABEL"],
        "COX_PARTIAL_HAZARD": risk_test,
        "RISK_GROUP": risk_group,
    })
    predictions.sort_values("PATIENT_ID").to_csv(output_dir / "survival_test_predictions.csv", index=False)

    summary = cox.summary.reset_index().rename(columns={"covariate": "FEATURE"})
    keep = [c for c in ["FEATURE", "coef", "exp(coef)", "se(coef)", "coef lower 95%", "coef upper 95%", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"] if c in summary]
    summary[keep].sort_values("p").to_csv(output_dir / "cox_coefficients.csv", index=False)
    joblib.dump(preprocess, output_dir / "survival_preprocessor.joblib")
    joblib.dump(cox, output_dir / "cox_model.pkl")
    save_curves(outcome, endpoint, horizon, output_dir)

    counts = outcome["EVENT_CODE"].value_counts().to_dict()
    metrics = {
        "endpoint": endpoint,
        "horizon_days": horizon,
        "n_patients": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "primary_events": int(counts.get(1, 0)),
        "competing_deaths": int(counts.get(2, 0)),
        "censored": int(counts.get(0, 0)),
        "test_concordance_index": float(c_index),
        "cox_penalizer": penalizer,
        "risk_group_cutoffs_from_training_partial_hazard": {
            "low_upper": float(q1), "intermediate_upper": float(q2)
        },
        "baseline_predictor_columns": available,
        "unavailable_predictors": sorted(set(BASELINE_FEATURES) - set(available)),
        "interpretation_note": (
            "For HF hospitalization, the Cox model is cause-specific: deaths before "
            "hospitalization are censored at death in the model, while Aalen-Johansen "
            "estimates cumulative incidence with death as a competing event."
        ),
    }
    (output_dir / "survival_analysis_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Survival analysis for the HFpEF cohort.")
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sheet-name", default="Synthetic SQL Output")
    parser.add_argument(
        "--endpoint", choices=["hf_hospitalization", "death", "composite"],
        default="hf_hospitalization",
    )
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--output-dir", type=Path, default=Path("survival_results"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--penalizer", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.horizon_days < 1:
        parser.error("--horizon-days must be positive.")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")
    if args.penalizer < 0:
        parser.error("--penalizer cannot be negative.")
    return args


def main() -> None:
    args = parse_args()
    df = add_age_at_index(load_workbook(args.input_xlsx, args.sheet_name))
    metrics = run_analysis(
        df, args.endpoint, args.horizon_days, args.output_dir,
        args.test_size, args.penalizer, args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
