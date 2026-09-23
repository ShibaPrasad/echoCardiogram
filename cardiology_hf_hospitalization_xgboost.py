#!/usr/bin/env python3
"""XGBoost model for index-date prediction of 365-day HF hospitalization.

The default outcome is the first inpatient hospitalization with a primary
heart-failure diagnosis on days 1-365 after the HFpEF index date. Rows with an
indeterminate hospitalization outcome (early loss to follow-up or death before
hospitalization) are excluded from binary training rather than labeled as 0.

Only index-date/baseline variables are eligible predictors. Post-index event,
death, censoring, and follow-up columns are retained for outcome auditing but
are never provided to XGBoost. All results from the supplied workbook remain
synthetic and are not clinical performance estimates.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is required. Install dependencies with: "
        "pip install xgboost pandas scikit-learn openpyxl matplotlib joblib"
    ) from exc


RANDOM_SEED = 42
DEFAULT_TARGET = "HF_HOSPITALIZATION_WITHIN_365D"
SYNTHETIC_TARGET = "SYNTHETIC_HF_HOSPITALIZATION_WITHIN_365D"
DEFAULT_DECISION_THRESHOLD = 0.20
DEFAULT_LOW_RISK_CUTOFF = 0.10
DEFAULT_HIGH_RISK_CUTOFF = 0.20
SUPPORTED_OUTCOMES = {
    "HF_HOSPITALIZATION_WITHIN_365D",
    "DEATH_WITHIN_365D",
    "HF_HOSPITALIZATION_OR_DEATH_WITHIN_365D",
}
SCRIPT_DIR = Path(__file__).resolve().parent
KNOWN_WORKBOOK_NAMES = (
    "synthetic_cardiology_hfpef_cohort.xlsx",
    "synthetic_cardiology_dataset_sql_aligned.xlsx",
)
DEFAULT_INPUT = SCRIPT_DIR / KNOWN_WORKBOOK_NAMES[0]

# Explicit allow-list: only information available on/before the HFpEF index date.
BASELINE_FEATURES = [
    "AGE_AT_INDEX", "SEX", "ETHNICITY", "RACE", "GENDER_IDENTITY",
    "SEX_ASSIGNED_AT_BIRTH", "INTERPRETER_NEEDED_YN",
    "INDEX_HFPEF_DX_CODE", "INDEX_HFPEF_DX_CODE_TYPE",
    "PRIMARY_DIAGNOSIS_YN", "PRESENT_ON_ADMISSION", "ED_DIAGNOSIS_YN",
    "CHRONIC_DIAGNOSIS_YN", "SERVICE_DEPARTMENT_NAME", "SERVICE_CENTER",
    "DISTINCT_TTE_ACCESSION_COUNT_IN_90D_BASELINE", "DAYS_TTE_BEFORE_INDEX",
    "BASELINE_LVEF", "BASELINE_E_PRIME", "BASELINE_E_OVER_E_PRIME",
    "BASELINE_TR_VELOCITY", "BASELINE_LAVI", "BASELINE_GLS",
    "BASELINE_SEPTAL_WALL_THICKNESS", "BASELINE_POSTERIOR_WALL_THICKNESS",
    "BASELINE_LV_MASS", "BASELINE_LV_MASS_INDEX",
    "HAS_SGLT2_ACTIVE_AT_INDEX", "HAS_INCRETIN_ACTIVE_AT_INDEX",
    "MEDICATION_EXPOSURE_GROUP", "HAS_ANTI_OBESITY_ACTIVE_AT_INDEX",
    "ACTIVE_MEDICATION_RECORD_COUNT_AT_INDEX",
    "DAYS_MEDICATION_STARTED_BEFORE_INDEX", "MEDICATION_CATEGORY",
    "MEDICATION_ORDER_STATUS", "GENERIC_NAME", "DOSE_AMT", "DOSE_UNIT",
    "FREQUENCY", "ROUTE", "ACTIVE_ORDER",
    "BMI_30_RECORD_COUNT_IN_365D_BASELINE", "MIN_BMI_GE_30",
    "MAX_BMI_GE_30", "AVG_BMI_GE_30", "DAYS_BMI_BEFORE_INDEX",
    "BASELINE_BMI", "BMI_RECORD_HEIGHT", "BMI_RECORD_WEIGHT",
    "DAYS_VITAL_BEFORE_INDEX", "BLOOD_PRESSURE_SYSTOLIC",
    "BLOOD_PRESSURE_DIASTOLIC", "PULSE", "PULSE_OXYMETRY",
    "RESPIRATORY_RATE", "BODY_TEMPERATURE", "BODY_SURFACE_AREA",
    "LAB_RESULT_COUNT_IN_90D_BASELINE",
    "DISTINCT_LAB_LOINC_COUNT_IN_90D_BASELINE", "DAYS_LAB_BEFORE_INDEX",
    "BASELINE_PROBNP_RESULT_VALUE_NUM", "BASELINE_HBA1C", "BASELINE_EGFR",
    "PREINDEX_DIAGNOSIS_RECORD_COUNT", "PREINDEX_PROBLEM_RECORD_COUNT",
    "PREINDEX_MEDICAL_HISTORY_RECORD_COUNT",
    # Structured ECG recorded in the 90-day baseline window.
    "DAYS_ECG_BEFORE_INDEX", "ECG_COUNT_IN_90D_BASELINE", "ECG_HEART_RATE_BPM",
    "ECG_PR_INTERVAL_MS", "ECG_QRS_DURATION_MS", "ECG_QT_INTERVAL_MS",
    "ECG_QTC_INTERVAL_MS", "ECG_P_AXIS_DEG", "ECG_QRS_AXIS_DEG", "ECG_T_AXIS_DEG",
    "ECG_ATRIAL_FIBRILLATION_FLAG", "ECG_ATRIAL_FLUTTER_FLAG", "ECG_LBBB_FLAG",
    "ECG_RBBB_FLAG", "ECG_AV_BLOCK_FLAG", "ECG_LVH_FLAG",
    "ECG_PACED_RHYTHM_FLAG", "ECG_ABNORMAL_FLAG",
    # Additional structured echo measurements; raw images/videos are excluded.
    "BASELINE_MITRAL_E_VELOCITY", "BASELINE_MITRAL_A_VELOCITY", "BASELINE_E_A_RATIO",
    "BASELINE_SEPTAL_E_PRIME", "BASELINE_LATERAL_E_PRIME", "BASELINE_AVERAGE_E_PRIME",
    "BASELINE_DECELERATION_TIME_MS", "BASELINE_DIASTOLIC_FUNCTION_GRADE",
    "BASELINE_PASP_MMHG", "BASELINE_TAPSE_CM", "BASELINE_RV_S_PRIME_CM_S",
    "BASELINE_RV_FAC_PERCENT", "BASELINE_RV_FUNCTION_GRADE",
    "BASELINE_MITRAL_REGURGITATION_GRADE", "BASELINE_TRICUSPID_REGURGITATION_GRADE",
    "BASELINE_IVC_DIAMETER_CM", "BASELINE_IVC_COLLAPSIBILITY_PERCENT",
    # Healthcare utilization during the 365 days before the index date.
    "PRIOR_HF_HOSPITALIZATION_COUNT_365D", "PRIOR_ALL_CAUSE_HOSPITALIZATION_COUNT_365D",
    "PRIOR_ED_VISIT_COUNT_365D", "PRIOR_CARDIOLOGY_VISIT_COUNT_365D",
    "DAYS_SINCE_LAST_HF_HOSPITALIZATION", "DAYS_SINCE_LAST_ED_VISIT",
    "PRIOR_TOTAL_INPATIENT_DAYS_365D", "PRIOR_ICU_ADMISSION_FLAG",
    # Explicit pre-index comorbidity indicators and summary scores.
    "ATRIAL_FIBRILLATION_FLAG", "HYPERTENSION_FLAG", "TYPE2_DIABETES_FLAG",
    "CHRONIC_KIDNEY_DISEASE_FLAG", "CKD_STAGE", "CORONARY_ARTERY_DISEASE_FLAG",
    "PRIOR_MI_FLAG", "PRIOR_STROKE_FLAG", "COPD_FLAG", "SLEEP_APNEA_FLAG",
    "PULMONARY_HYPERTENSION_FLAG", "ANEMIA_FLAG", "CHARLSON_COMORBIDITY_INDEX",
    "ELIXHAUSER_SCORE",
    # Expanded laboratory values and longitudinal 90-day summaries.
    "BASELINE_CREATININE_MG_DL", "BASELINE_BUN_MG_DL", "BASELINE_SODIUM_MMOL_L",
    "BASELINE_POTASSIUM_MMOL_L", "BASELINE_HEMOGLOBIN_G_DL",
    "BASELINE_HEMATOCRIT_PERCENT", "BASELINE_ALBUMIN_G_DL",
    "BASELINE_GLUCOSE_MG_DL", "BASELINE_TROPONIN_NG_L", "EGFR_MEAN_90D",
    "EGFR_MIN_90D", "EGFR_SLOPE_90D", "EGFR_MEASUREMENT_COUNT_90D",
    "PROBNP_MEAN_90D", "PROBNP_MAX_90D", "PROBNP_SLOPE_90D",
    "PROBNP_MEASUREMENT_COUNT_90D",
    # Other cardiovascular therapies active on the index date.
    "LOOP_DIURETIC_ACTIVE_AT_INDEX", "LOOP_DIURETIC_FUROSEMIDE_EQUIVALENT_MG_DAY",
    "MRA_ACTIVE_AT_INDEX", "ACEI_ARB_ARNI_ACTIVE_AT_INDEX",
    "BETA_BLOCKER_ACTIVE_AT_INDEX", "CALCIUM_CHANNEL_BLOCKER_ACTIVE_AT_INDEX",
    "ANTICOAGULANT_ACTIVE_AT_INDEX", "ANTIPLATELET_ACTIVE_AT_INDEX",
    "STATIN_ACTIVE_AT_INDEX", "TOTAL_ACTIVE_MEDICATION_COUNT_AT_INDEX",
    # Longitudinal BMI and vital summaries.
    "BMI_SLOPE_365D", "BMI_VARIABILITY_365D", "SBP_MEAN_90D", "SBP_MAX_90D",
    "SBP_SLOPE_90D", "VITAL_MEASUREMENT_COUNT_90D",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def assign_risk_groups(probability: np.ndarray, low_cutoff: float, high_cutoff: float) -> np.ndarray:
    """Convert continuous predicted risks into ordered research risk groups."""
    return np.select(
        [probability < low_cutoff, probability < high_cutoff],
        ["Low", "Intermediate"],
        default="High",
    )


def resolve_input_workbook(path: Path) -> Path:
    """Resolve an Excel input independently of the shell working directory."""
    requested = path.expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(SCRIPT_DIR / requested)
    candidates.extend(SCRIPT_DIR / name for name in KNOWN_WORKBOOK_NAMES)

    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in checked:
            checked.append(candidate)
        if candidate.is_file():
            return candidate

    nearby_excel = sorted(file.name for file in SCRIPT_DIR.glob("*.xlsx"))
    raise FileNotFoundError(
        "Input workbook was not found. Checked:\n  - "
        + "\n  - ".join(str(candidate) for candidate in checked)
        + f"\nExcel files beside the Python script: {nearby_excel or 'none'}"
        + "\nYou can also provide the full path with --input-xlsx /path/to/file.xlsx"
    )


def load_sql_aligned_workbook(path: Path, sheet_name: str) -> pd.DataFrame:
    resolved_path = resolve_input_workbook(path)
    print(f"Loading workbook: {resolved_path}")
    excel = pd.ExcelFile(resolved_path)
    selected_sheet = sheet_name
    if selected_sheet not in excel.sheet_names:
        if sheet_name == "Synthetic SQL Output":
            selected_sheet = excel.sheet_names[0]
            warnings.warn(
                f"Sheet '{sheet_name}' was not found; using first sheet "
                f"'{selected_sheet}'. Available sheets: {excel.sheet_names}",
                RuntimeWarning, stacklevel=2,
            )
        else:
            raise ValueError(
                f"Sheet '{sheet_name}' was not found. Available sheets: "
                f"{excel.sheet_names}"
            )
    print(f"Loading sheet: {selected_sheet}")
    df = pd.read_excel(excel, sheet_name=selected_sheet)
    df.columns = [str(column).strip().upper() for column in df.columns]
    if df.empty:
        raise ValueError("The selected Excel sheet contains no patient rows.")
    return df


def add_age_at_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    birth = pd.to_datetime(result["BIRTH_DATE"], errors="coerce")
    index = pd.to_datetime(result["FIRST_RECORDED_HFPEF_INDEX_DATE"], errors="coerce")
    result["AGE_AT_INDEX"] = ((index - birth).dt.days / 365.25).round(1)
    return result


def validate_sql_cohort(df: pd.DataFrame) -> dict:
    required = {
        "PATIENT_ID", "FIRST_RECORDED_HFPEF_INDEX_DATE", "INDEX_HFPEF_DX_CODE",
        "BASELINE_BMI", "BASELINE_BMI_COLLECTION_DATETIME",
        "HAS_SGLT2_ACTIVE_AT_INDEX", "HAS_INCRETIN_ACTIVE_AT_INDEX",
        "MEDICATION_START_DATE",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Workbook is missing required SQL-output columns: {missing}")

    index = pd.to_datetime(df["FIRST_RECORDED_HFPEF_INDEX_DATE"], errors="coerce")
    bmi_date = pd.to_datetime(df["BASELINE_BMI_COLLECTION_DATETIME"], errors="coerce")
    med_start = pd.to_datetime(df["MEDICATION_START_DATE"], errors="coerce")
    diagnosis_ok = df["INDEX_HFPEF_DX_CODE"].astype("string").str.replace(
        ".", "", regex=False
    ).str.match(r"^(I503|4283)", na=False)
    medication_ok = (
        pd.to_numeric(df["HAS_SGLT2_ACTIVE_AT_INDEX"], errors="coerce").fillna(0).eq(1)
        | pd.to_numeric(df["HAS_INCRETIN_ACTIVE_AT_INDEX"], errors="coerce").fillna(0).eq(1)
    )
    bmi_ok = pd.to_numeric(df["BASELINE_BMI"], errors="coerce").ge(30)
    bmi_window_ok = bmi_date.between(index - pd.Timedelta(days=365), index)
    med_timing_ok = med_start.le(index)
    failures = {
        "missing_index_date": int(index.isna().sum()),
        "invalid_hfpef_code": int((~diagnosis_ok).sum()),
        "bmi_below_30_or_missing": int((~bmi_ok).sum()),
        "bmi_outside_365d_window": int((~bmi_window_ok).sum()),
        "no_qualifying_medication_at_index": int((~medication_ok).sum()),
        "medication_starts_after_index": int((~med_timing_ok).sum()),
        "duplicate_patient_id": int(df["PATIENT_ID"].duplicated().sum()),
    }
    if any(failures.values()):
        raise ValueError(f"SQL cohort validation failed: {failures}")
    return {"n_patients": int(len(df)), **failures}

# generate_demo_target() not needed in our case as the workbook already contains explicit outcome labels. 
# if lable not provided in data file - then this function artificially creates a hospitalization label from selected baseline variables

def generate_demo_target(df: pd.DataFrame, seed: int) -> pd.Series:
    """Create an artificial label strictly for pipeline testing."""
    rng = np.random.default_rng(seed)

    def number(name: str, default: float) -> np.ndarray:
        values = pd.to_numeric(df.get(name), errors="coerce")
        fill = values.median() if values.notna().any() else default
        return values.fillna(fill).to_numpy()

    age = number("AGE_AT_INDEX", 68)
    bmi = number("BASELINE_BMI", 35)
    probnp = np.maximum(number("BASELINE_PROBNP_RESULT_VALUE_NUM", 350), 0)
    egfr = number("BASELINE_EGFR", 65)
    e_over_e = number("BASELINE_E_OVER_E_PRIME", 13)
    tr_velocity = number("BASELINE_TR_VELOCITY", 2.6)
    lavi = number("BASELINE_LAVI", 36)
    diagnoses = number("PREINDEX_DIAGNOSIS_RECORD_COUNT", 8)
    sglt2 = number("HAS_SGLT2_ACTIVE_AT_INDEX", 0)
    incretin = number("HAS_INCRETIN_ACTIVE_AT_INDEX", 0)
    log_odds = (
        -3.30 + 0.028 * (age - 65) + 0.045 * (bmi - 30)
        + 0.42 * np.log1p(probnp / 100) - 0.012 * (egfr - 60)
        + 0.070 * (e_over_e - 10) + 0.35 * (tr_velocity > 2.8)
        + 0.010 * (lavi - 34) + 0.020 * diagnoses
        - 0.18 * sglt2 - 0.08 * incretin
        + rng.normal(0, 0.35, len(df))
    )
    probability = np.clip(sigmoid(log_odds), 0.01, 0.90)
    return pd.Series(rng.binomial(1, probability), index=df.index, name=SYNTHETIC_TARGET)


def resolve_target(
    df: pd.DataFrame, target_column: str, generate_synthetic: bool, seed: int
) -> tuple[pd.DataFrame, str, str, dict]:
    target = target_column.strip().upper()
    result = df.copy()
    if target in result.columns:
        source = "workbook-supplied outcome label"
    elif generate_synthetic:
        result[SYNTHETIC_TARGET] = generate_demo_target(result, seed)
        target = SYNTHETIC_TARGET
        source = "synthetic demonstration target generated from baseline variables"
        warnings.warn(
            "Using an artificial hospitalization target for pipeline testing only. "
            "Metrics are not clinical performance estimates.", RuntimeWarning, stacklevel=2
        )
    else:
        raise ValueError(
            f"Target column '{target}' was not found. Add an encounter/mortality-"
            "derived binary outcome, or use --generate-synthetic-target only for "
            "software testing."
        )
    y = pd.to_numeric(result[target], errors="coerce")
    invalid = y.notna() & ~y.isin([0, 1])
    if invalid.any():
        raise ValueError(f"Target '{target}' contains values other than 0, 1, or blank.")
    target_summary = {
        "n_rows_before_target_filter": int(len(result)),
        "n_indeterminate_excluded": int(y.isna().sum()),
        "n_events": int(y.eq(1).sum()),
        "n_nonevents": int(y.eq(0).sum()),
    }
    result = result.loc[y.notna()].copy()
    result[target] = y.loc[y.notna()].astype(int)
    if result[target].nunique() < 2:
        raise ValueError(f"Target '{target}' must contain both 0 and 1 after excluding blanks.")
    target_summary["n_rows_used_for_model"] = int(len(result))
    if target in SUPPORTED_OUTCOMES:
        source += "; derived from the workbook's 365-day follow-up definition"
    return result, target, source, target_summary


def select_baseline_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    available = [column for column in BASELINE_FEATURES if column in df.columns]
    missing = [column for column in BASELINE_FEATURES if column not in df.columns]
    all_missing = [column for column in available if df[column].isna().all()]
    selected = [column for column in available if column not in all_missing]
    if not selected:
        raise ValueError("None of the expected SQL-aligned baseline features were found.")
    return df[selected].copy(), sorted(set(missing + all_missing))


def identify_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Pandas 2/3/4-compatible selection without the Pandas4Warning."""
    categorical = [
        column for column in X.columns
        if is_object_dtype(X[column].dtype)
        or is_string_dtype(X[column].dtype)
        or isinstance(X[column].dtype, pd.CategoricalDtype)
    ]
    return categorical, [column for column in X.columns if column not in categorical]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def train_and_evaluate(
    df: pd.DataFrame, target: str, target_source: str, output_dir: Path,
    test_size: float, threshold: float, seed: int, n_jobs: int, validation: dict,
    target_summary: dict, low_risk_cutoff: float, high_risk_cutoff: float,
) -> dict:
    X, unavailable = select_baseline_features(df)
    y = df[target].astype(int)
    patient_ids = df["PATIENT_ID"].astype(str)
    categorical, numeric = identify_column_types(X)
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]), categorical),
    ], remainder="drop", verbose_feature_names_out=True)

    X_train, X_test, y_train, y_test, _, id_test = train_test_split(
        X, y, patient_ids, test_size=test_size, stratify=y, random_state=seed
    )
    negative, positive = int((y_train == 0).sum()), int((y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=450, max_depth=4, learning_rate=0.03, min_child_weight=4,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.15, reg_lambda=1.5,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=negative / max(positive, 1), tree_method="hist",
        random_state=seed, n_jobs=n_jobs,
    )
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    risk_group = assign_risk_groups(probability, low_risk_cutoff, high_risk_cutoff)
    cm = confusion_matrix(y_test, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    risk_frame = pd.DataFrame({"RISK_GROUP": risk_group, "OBSERVED_EVENT": y_test.to_numpy()})
    risk_order = ["Low", "Intermediate", "High"]
    risk_summary_frame = (
        risk_frame.groupby("RISK_GROUP", observed=False)["OBSERVED_EVENT"]
        .agg(PATIENT_COUNT="size", EVENT_COUNT="sum", OBSERVED_EVENT_RATE="mean")
        .reindex(risk_order, fill_value=0)
        .reset_index()
    )
    risk_summary = [
        {
            "risk_group": str(row.RISK_GROUP),
            "patient_count": int(row.PATIENT_COUNT),
            "event_count": int(row.EVENT_COUNT),
            "observed_event_rate": float(row.OBSERVED_EVENT_RATE),
        }
        for row in risk_summary_frame.itertuples(index=False)
    ]
    metrics = {
        "target_column": target, "target_source": target_source,
        "synthetic_target_warning": target == SYNTHETIC_TARGET,
        "n_total": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "event_rate_total": float(y.mean()), "decision_threshold": threshold,
        "risk_stratification_cutoffs": {
            "low": f"probability < {low_risk_cutoff:.3f}",
            "intermediate": f"{low_risk_cutoff:.3f} <= probability < {high_risk_cutoff:.3f}",
            "high": f"probability >= {high_risk_cutoff:.3f}",
        },
        "test_set_risk_stratification": risk_summary,
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "average_precision_pr_auc": float(average_precision_score(y_test, probability)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_test, prediction, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, probability)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(y_test, prediction, output_dict=True, zero_division=0),
        "target_filtering": target_summary,
        "baseline_predictor_columns": X.columns.tolist(),
        "unavailable_or_all_missing_features": unavailable,
        "cohort_validation": validation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "xgboost_metrics.json").write_text(json.dumps(metrics, indent=2))
    prediction_output = pd.DataFrame({
        "PATIENT_ID": id_test.to_numpy(),
        f"OBSERVED_{target}": y_test.to_numpy(),
        "PREDICTED_PROBABILITY": probability,
        f"PREDICTED_CLASS_AT_{threshold:.2f}": prediction,
        "RISK_GROUP": risk_group,
    })
    audit_columns = [
        "PATIENT_ID", "FIRST_HF_HOSPITALIZATION_DATETIME",
        "DAYS_INDEX_TO_FIRST_HF_HOSPITALIZATION", "DEATH_WITHIN_365D",
        "DAYS_INDEX_TO_DEATH", "FIRST_EVENT_WITHIN_365D", "CENSORING_REASON",
    ]
    available_audit = [column for column in audit_columns if column in df.columns]
    if len(available_audit) > 1:
        prediction_output = prediction_output.merge(
            df[available_audit].drop_duplicates("PATIENT_ID"), on="PATIENT_ID", how="left"
        )
    prediction_output.sort_values("PATIENT_ID").to_csv(
        output_dir / "xgboost_test_predictions.csv", index=False
    )
    risk_summary_frame.to_csv(output_dir / "xgboost_risk_stratification.csv", index=False)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pipeline.named_steps["model"].feature_importances_
    pd.DataFrame({"FEATURE": names, "IMPORTANCE": importance}).sort_values(
        "IMPORTANCE", ascending=False
    ).to_csv(output_dir / "xgboost_feature_importance.csv", index=False)
    joblib.dump(pipeline, output_dir / "xgboost_pipeline.joblib")

    fpr, tpr, _ = roc_curve(y_test, probability)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, probability)
    cal_true, cal_pred = calibration_curve(y_test, probability, n_bins=8, strategy="quantile")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].imshow(cm, cmap="Blues")
    for (row, col), value in np.ndenumerate(cm):
        axes[0, 0].text(col, row, str(value), ha="center", va="center", fontsize=12)
    axes[0, 0].set(title=f"Confusion matrix (threshold {threshold:.2f})", xlabel="Predicted", ylabel="Observed", xticks=[0, 1], yticks=[0, 1])
    axes[0, 1].plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 1].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0, 1].legend(loc="lower right")
    axes[1, 0].plot(pr_recall, pr_precision, label=f"PR AUC = {metrics['average_precision_pr_auc']:.3f}")
    axes[1, 0].axhline(y.mean(), linestyle="--", color="gray", label="Event prevalence")
    axes[1, 0].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    axes[1, 0].legend(loc="upper right")
    axes[1, 1].plot(cal_pred, cal_true, marker="o", label="Model")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[1, 1].set(title="Calibration", xlabel="Mean predicted probability", ylabel="Observed event rate")
    axes[1, 1].legend(loc="upper left")
    fig.suptitle(f"XGBoost evaluation: {target}")
    fig.tight_layout()
    fig.savefig(output_dir / "xgboost_evaluation.png", dpi=180)
    plt.close(fig)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost using the SQL-aligned cardiology Excel cohort.")
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sheet-name", default="Synthetic SQL Output")
    parser.add_argument(
        "--target-column", default=DEFAULT_TARGET,
        help=("Outcome column. Recommended: HF_HOSPITALIZATION_WITHIN_365D; "
              "alternatives: DEATH_WITHIN_365D or "
              "HF_HOSPITALIZATION_OR_DEATH_WITHIN_365D."),
    )
    parser.add_argument("--generate-synthetic-target", action="store_true", help="Generate an artificial label for testing only when the target is absent.")
    parser.add_argument("--output-dir", type=Path, default=Path("cardiology_xgboost_sql_outputs"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_DECISION_THRESHOLD,
        help="Binary classification threshold; default 0.20.",
    )
    parser.add_argument(
        "--low-risk-cutoff", type=float, default=DEFAULT_LOW_RISK_CUTOFF,
        help="Upper boundary for the Low risk group; default 0.10.",
    )
    parser.add_argument(
        "--high-risk-cutoff", type=float, default=DEFAULT_HIGH_RISK_CUTOFF,
        help="Lower boundary for the High risk group; default 0.20.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1.")
    if not 0 <= args.low_risk_cutoff < args.high_risk_cutoff <= 1:
        parser.error("Risk cutoffs must satisfy 0 <= low < high <= 1.")
    return args


def main() -> None:
    args = parse_args()
    df = add_age_at_index(load_sql_aligned_workbook(args.input_xlsx, args.sheet_name))
    validation = validate_sql_cohort(df)
    df, target, source, target_summary = resolve_target(
        df, args.target_column, args.generate_synthetic_target, args.seed
    )
    metrics = train_and_evaluate(
        df, target, source, args.output_dir, args.test_size, args.threshold,
        args.seed, args.n_jobs, validation, target_summary,
        args.low_risk_cutoff, args.high_risk_cutoff,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
