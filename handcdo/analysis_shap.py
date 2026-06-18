from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_analysis(input_csv: str | Path, output_dir: str | Path, target: str = "hand_score") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    df = df.copy()

    if "failed" in df.columns:
        failed_mask = df["failed"].astype(str).str.strip().str.lower().isin(
            {"1", "true", "t", "yes", "y"}
        )
        df = df[~failed_mask].copy()

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    if df.empty:
        raise ValueError("No successful rows available for analysis")

    y = df[target].astype(float)

    drop_exact = {
        "design_id",
        target,
        "best_available_score",
        "failed",
        "error",
        "fidelity",
    }
    drop_prefixes = (
        "failed_",
        "error_",
        "backend_",
        "config_path_",
        "n_grasp_trials_",
        "sampler_",
        "seed_",
        "hand_score_",
    )

    def keep_feature(column: str) -> bool:
        if column in drop_exact:
            return False
        if column.endswith("_best_score") or "_best_score_" in column:
            return False
        if any(column.startswith(prefix) for prefix in drop_prefixes):
            return False
        return True

    feature_df = df[[c for c in df.columns if keep_feature(c)]]
    cat_cols = [c for c in feature_df.columns if not pd.api.types.is_numeric_dtype(feature_df[c])]
    num_cols = [c for c in feature_df.columns if c not in cat_cols]
    feature_df = feature_df.copy()
    feature_df[cat_cols] = feature_df[cat_cols].fillna("missing").astype(str)
    feature_df[num_cols] = feature_df[num_cols].fillna(feature_df[num_cols].median(numeric_only=True))
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols), ("num", "passthrough", num_cols)]
    )
    model = RandomForestRegressor(n_estimators=300, random_state=0, min_samples_leaf=1)
    pipe = Pipeline([("pre", pre), ("rf", model)])
    pipe.fit(feature_df, y)
    feature_names = list(pipe.named_steps["pre"].get_feature_names_out())
    importances = pipe.named_steps["rf"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "rf_importance": importances}).sort_values("rf_importance", ascending=False)
    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(y)), y.cummax(), marker="o", linewidth=1)
    plt.xlabel("completed evaluation")
    plt.ylabel("best hand score")
    plt.tight_layout()
    plt.savefig(output_dir / "optimization_convergence.png", dpi=160)
    plt.close()
    best_row = df.loc[y.idxmax()].to_dict()
    (output_dir / "best_design.json").write_text(json.dumps(best_row, indent=2, sort_keys=True), encoding="utf-8")
    try:
        import shap

        x_enc = pipe.named_steps["pre"].transform(feature_df)
        explainer = shap.TreeExplainer(pipe.named_steps["rf"])
        shap_values = explainer.shap_values(x_enc)
        plt.figure()
        shap.summary_plot(shap_values, x_enc, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()
        shap_mean = abs(shap_values).mean(axis=0)
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": shap_mean}).sort_values(
            "mean_abs_shap", ascending=False
        ).to_csv(output_dir / "shap_importance.csv", index=False)
    except Exception as exc:
        (output_dir / "shap_unavailable.txt").write_text(f"SHAP analysis skipped: {type(exc).__name__}: {exc}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="outputs/results.csv")
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--target", default="hand_score")
    args = parser.parse_args()
    run_analysis(args.input_csv, args.output_dir, args.target)


if __name__ == "__main__":
    main()
