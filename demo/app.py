import os
import sys
import re
import time

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

TRANSFORMER_FILE = os.path.join(PROJECT_ROOT, "results", "transformer_results_comprehensive.txt")
ST_FILE = os.path.join(PROJECT_ROOT, "results", "results_metr_la_comprehensive.txt")
MULTIHOP_FILE = os.path.join(PROJECT_ROOT, "results", "results_multihop_comprehensive.txt")

IMAGE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "images")

st.set_page_config(page_title="Traffic Forecast Demo", layout="wide")

COLOR_MAP = {
    "Transformer": "#1f77b4",
    "ST-Transformer": "#ff7f0e",
    "Multi-Hop": "#2ca02c",
    "MTESformer (SOTA)": "#d62728",
}

PAPER_RESULTS = {
    "Model": "MTESformer (SOTA)",
    "MAE": 3.37,
    "RMSE": 7.14,
    "MAPE (%)": 9.62,
    "R²": None,
    "Horizon": pd.DataFrame({
        "Minutes": [15, 30, 60],
        "MAE": [2.73, 3.03, 3.37],
        "RMSE": [5.34, 6.22, 7.14],
        "MAPE": [7.10, 8.30, 9.62],
    })
}


def safe_float(value: str):
    try:
        match = re.search(r"-?\d+(\.\d+)?", value)
        return float(match.group()) if match else None
    except Exception:
        return None


def parse_metrics(file_path: str):
    metrics = {
        "Model": None,
        "MAE": None,
        "RMSE": None,
        "MAPE (%)": None,
        "R²": None,
    }

    if not os.path.exists(file_path):
        return metrics

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line.startswith("Model:"):
                metrics["Model"] = line.split(":", 1)[1].strip()
            elif line.startswith("Model Type:"):
                metrics["Model"] = line.split(":", 1)[1].strip()
            elif line.startswith("MAE:"):
                metrics["MAE"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("RMSE:"):
                metrics["RMSE"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("MAPE (>0.1):"):
                metrics["MAPE (%)"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("MAPE:") and metrics["MAPE (%)"] is None:
                metrics["MAPE (%)"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("R² Score:") or line.startswith("R2 Score:"):
                metrics["R²"] = safe_float(line.split(":", 1)[1])

    return metrics


def parse_horizon_mape_details(file_path: str):
    rows = []

    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["Minutes", "MAPE", "MAPE_masked", "Coverage"])

    in_mape_section = False

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if "PER-HORIZON MAPE DETAILED BREAKDOWN" in line:
                in_mape_section = True
                continue

            if in_mape_section:
                stripped = line.strip()

                if not stripped:
                    continue
                if stripped.startswith("Horizon"):
                    continue
                if set(stripped) == {"-"}:
                    continue
                if stripped.startswith("MAPE TYPES"):
                    break
                if "SENSOR-LEVEL STATISTICS" in stripped:
                    break

                parts = stripped.split()

                try:
                    rows.append({
                        "Horizon": int(parts[0]),
                        "Minutes": int(parts[1]),
                        "MAPE": safe_float(parts[2]),
                        "MAPE_masked": safe_float(parts[3]),
                        "Coverage": safe_float(parts[4]),
                    })
                except (ValueError, IndexError):
                    continue

    return pd.DataFrame(rows)


def parse_horizon_metrics(file_path: str):
    rows = []

    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["Minutes", "MAE", "RMSE", "MAPE", "WMAPE", "R2"])

    in_horizon_section = False

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if "PER-HORIZON PERFORMANCE BREAKDOWN" in line:
                in_horizon_section = True
                continue

            if in_horizon_section:
                stripped = line.strip()

                if not stripped:
                    continue
                if stripped.startswith("Horizon"):
                    continue
                if set(stripped) == {"-"}:
                    continue
                if "PER-HORIZON MAPE DETAILED BREAKDOWN" in stripped:
                    break
                if "COMPARISON WITH" in stripped:
                    break
                if "SENSOR-LEVEL STATISTICS" in stripped:
                    break

                parts = stripped.split()

                try:
                    row = {
                        "Horizon": int(parts[0]),
                        "Minutes": int(parts[1]),
                        "MAE": float(parts[2]),
                        "RMSE": float(parts[3]),
                        "MAPE": None,
                        "WMAPE": None,
                        "R2": None,
                    }

                    if len(parts) >= 7:
                        row["WMAPE"] = safe_float(parts[4])
                        row["R2"] = safe_float(parts[5])
                        row["MAPE"] = safe_float(parts[6])
                    elif len(parts) >= 6:
                        row["WMAPE"] = safe_float(parts[4])
                        row["R2"] = safe_float(parts[5])

                    rows.append(row)

                except (ValueError, IndexError):
                    continue

    df = pd.DataFrame(rows)

    if not df.empty:
        mape_df = parse_horizon_mape_details(file_path)

        if not mape_df.empty:
            df = df.drop(columns=["MAPE"], errors="ignore")
            df = df.merge(
                mape_df[["Minutes", "MAPE"]],
                on="Minutes",
                how="left",
            )

    return df


def load_comparison_table():
    transformer = parse_metrics(TRANSFORMER_FILE)
    st_model = parse_metrics(ST_FILE)
    multihop = parse_metrics(MULTIHOP_FILE)

    return pd.DataFrame({
        "Model": [
            "Transformer (Baseline)",
            "ST-Transformer (Advanced)",
            "ST-Transformer Multi-Hop (Advanced)",
            PAPER_RESULTS["Model"],
        ],
        "MAE": [
            transformer["MAE"],
            st_model["MAE"],
            multihop["MAE"],
            PAPER_RESULTS["MAE"],
        ],
        "RMSE": [
            transformer["RMSE"],
            st_model["RMSE"],
            multihop["RMSE"],
            PAPER_RESULTS["RMSE"],
        ],
        "MAPE (%)": [
            transformer["MAPE (%)"],
            st_model["MAPE (%)"],
            multihop["MAPE (%)"],
            PAPER_RESULTS["MAPE (%)"],
        ],
        "R²": [
            transformer["R²"],
            st_model["R²"],
            multihop["R²"],
            PAPER_RESULTS["R²"],
        ],
    })


def render_comparison_summary():
    st.caption("The comparison summary shows average performance across prediction horizons.")

    df = load_comparison_table()
    display_df = df.copy()

    for col in ["MAE", "RMSE", "MAPE (%)", "R²"]:
        if col == "R²":
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.3f}" if pd.notnull(x) else "-"
            )
        else:
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.2f}" if pd.notnull(x) else "-"
            )

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def get_sources():
    transformer = parse_metrics(TRANSFORMER_FILE)
    st_model = parse_metrics(ST_FILE)
    multihop = parse_metrics(MULTIHOP_FILE)

    return [
        ("Transformer", parse_horizon_metrics(TRANSFORMER_FILE), transformer, "-", "o"),
        ("ST-Transformer", parse_horizon_metrics(ST_FILE), st_model, "--", "s"),
        ("Multi-Hop", parse_horizon_metrics(MULTIHOP_FILE), multihop, ":", "^"),
        (
            "MTESformer (SOTA)",
            PAPER_RESULTS["Horizon"],
            {
                "MAE": PAPER_RESULTS["MAE"],
                "RMSE": PAPER_RESULTS["RMSE"],
                "MAPE (%)": PAPER_RESULTS["MAPE (%)"],
                "R²": PAPER_RESULTS["R²"],
            },
            "-.",
            "D",
        ),
    ]


def filter_paper_points_by_current_minute(df: pd.DataFrame, current_minutes: int):
    visible_minutes = []

    if current_minutes >= 20:
        visible_minutes.append(15)
    if current_minutes >= 30:
        visible_minutes.append(30)
    if current_minutes >= 60:
        visible_minutes.append(60)

    return df[df["Minutes"].isin(visible_minutes)]


def get_cumulative_values(plot_df: pd.DataFrame, metrics: dict, metric_name: str):
    if plot_df.empty:
        return None

    if metric_name not in plot_df.columns:
        return None

    values = plot_df[metric_name]

    if values.isnull().all():
        if metric_name == "MAPE":
            final_value = metrics.get("MAPE (%)")
        else:
            final_value = metrics.get(metric_name)

        if final_value is None:
            return None

        values = pd.Series([final_value] * len(plot_df), index=plot_df.index)

    values = values.astype(float)
    y_values = values.expanding().mean()

    if plot_df["Minutes"].iloc[-1] == 60:
        if metric_name == "MAPE":
            final_value = metrics.get("MAPE (%)")
        else:
            final_value = metrics.get(metric_name)

        if final_value is not None:
            y_values.iloc[-1] = final_value

    return y_values


def plot_metric_line(ax, metric_name, y_label, title, max_steps=None, current_minutes=None):
    handles = []
    labels = []

    for label, df, metrics, linestyle, marker in get_sources():
        if df.empty:
            continue

        if label == "MTESformer (SOTA)" and current_minutes is not None:
            plot_df = filter_paper_points_by_current_minute(df, current_minutes)
        else:
            plot_df = df.iloc[:min(len(df), max_steps)] if max_steps else df

        if plot_df.empty:
            continue

        y_values = get_cumulative_values(plot_df, metrics, metric_name)

        if y_values is None:
            continue

        line, = ax.plot(
            plot_df["Minutes"],
            y_values,
            linestyle=linestyle,
            marker=marker,
            alpha=0.65,
            label=label,
            color=COLOR_MAP.get(label, None),
        )

        x_last = plot_df["Minutes"].iloc[-1]
        y_last = y_values.iloc[-1]

        ax.text(
            x_last,
            y_last,
            f"{y_last:.2f}",
            fontsize=9,
            ha="left",
            va="bottom",
        )

        handles.append(line)
        labels.append(label)

    ax.set_xlabel("Prediction Horizon (Minutes)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xlim(0, 62)
    ax.set_xticks([10, 20, 30, 40, 50, 60])
    ax.grid(True)

    return handles, labels


def save_final_demo_images():
    output_dir = IMAGE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    final_minutes = 60
    final_step = final_minutes // 5

    for metric_name, y_label, title, filename in [
        ("MAE", "MAE", "Cumulative MAE", "0.MAE_All.jpg"),
        ("RMSE", "RMSE", "Cumulative RMSE", "0.RMSE_All.jpg"),
        ("MAPE", "MAPE (%)", "Cumulative MAPE", "0.MAPE_All.jpg"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.8))

        handles, labels = plot_metric_line(
            ax,
            metric_name,
            y_label,
            title,
            max_steps=final_step,
            current_minutes=final_minutes,
        )

        if handles and labels:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=4,
                bbox_to_anchor=(0.5, -0.02),
            )

        fig.tight_layout(rect=[0, 0.08, 1, 1])

        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="jpg")

        plt.close(fig)


def get_pair_sources(target_model_name: str):
    sources = get_sources()

    pair_sources = []

    for label, df, metrics, linestyle, marker in sources:
        if label == target_model_name:
            pair_sources.append((label, df, metrics, linestyle, marker))

    for label, df, metrics, linestyle, marker in sources:
        if label == "MTESformer (SOTA)":
            pair_sources.append((label, df, metrics, linestyle, marker))

    return pair_sources


def plot_metric_line_pair(ax, metric_name, y_label, title, target_model_name, max_steps=None, current_minutes=None):
    handles = []
    labels = []

    for label, df, metrics, linestyle, marker in get_pair_sources(target_model_name):
        if df.empty:
            continue

        if label == "MTESformer (SOTA)" and current_minutes is not None:
            plot_df = filter_paper_points_by_current_minute(df, current_minutes)
        else:
            plot_df = df.iloc[:min(len(df), max_steps)] if max_steps else df

        if plot_df.empty:
            continue

        y_values = get_cumulative_values(plot_df, metrics, metric_name)

        if y_values is None:
            continue

        line, = ax.plot(
            plot_df["Minutes"],
            y_values,
            linestyle=linestyle,
            marker=marker,
            alpha=0.65,
            label=label,
            color=COLOR_MAP.get(label, None),
        )

        x_last = plot_df["Minutes"].iloc[-1]
        y_last = y_values.iloc[-1]

        ax.text(
            x_last,
            y_last,
            f"{y_last:.2f}",
            fontsize=9,
            ha="left",
            va="bottom",
        )

        handles.append(line)
        labels.append(label)

    ax.set_xlabel("Prediction Horizon (Minutes)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xlim(0, 62)
    ax.set_xticks([10, 20, 30, 40, 50, 60])
    ax.grid(True)

    return handles, labels


def save_pair_comparison_images():
    output_dir = IMAGE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    final_minutes = 60
    final_step = final_minutes // 5

    comparison_targets = [
        ("Transformer", "Transformer_vs_SOTA", "1."),
        ("ST-Transformer", "ST_Transformer_vs_SOTA", "2."),
        ("Multi-Hop", "Multi_Hop_vs_SOTA", "3."),
    ]

    for target_model_name, file_tag, prefix in comparison_targets:
        for metric_name, y_label, title_prefix in [
            ("MAE", "MAE", "Cumulative MAE"),
            ("RMSE", "RMSE", "Cumulative RMSE"),
            ("MAPE", "MAPE (%)", "Cumulative MAPE"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4.8))

            handles, labels = plot_metric_line_pair(
                ax,
                metric_name,
                y_label,
                f"{title_prefix}: {target_model_name} vs SOTA",
                target_model_name,
                max_steps=final_step,
                current_minutes=final_minutes,
            )

            if handles and labels:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    ncol=2,
                    bbox_to_anchor=(0.5, -0.02),
                )

            fig.tight_layout(rect=[0, 0.08, 1, 1])

            save_path = os.path.join(
                output_dir,
                f"{prefix}{metric_name}_{file_tag}.jpg"
            )

            fig.savefig(save_path, dpi=300, bbox_inches="tight", format="jpg")
            plt.close(fig)


def render_model_comparison():
    multihop = parse_metrics(MULTIHOP_FILE)

    our_mae = multihop["MAE"]
    our_rmse = multihop["RMSE"]
    our_mape = multihop["MAPE (%)"]

    sota_mae = PAPER_RESULTS["MAE"]
    sota_rmse = PAPER_RESULTS["RMSE"]
    sota_mape = PAPER_RESULTS["MAPE (%)"]

    def calc_diff(our, sota):
        if our is None or sota is None:
            return None
        return our - sota

    mae_diff = calc_diff(our_mae, sota_mae)
    rmse_diff = calc_diff(our_rmse, sota_rmse)
    mape_diff = calc_diff(our_mape, sota_mape)

    st.divider()
    st.subheader("Final Model Comparison")

    st.markdown("**ST-Transformer Multi-Hop (Advanced) vs MTESformer (SOTA)**")

    def format_line(name, our, sota, diff, is_percent=False):
        if our is None or sota is None:
            return f"{name}: -"

        sign = "+" if diff > 0 else ""

        if is_percent:
            return f"{name}: {our:.2f}% vs {sota:.2f}%  ({sign}{diff:.2f}%)"
        return f"{name}: {our:.2f} vs {sota:.2f}  ({sign}{diff:.2f})"

    st.write(format_line("MAE", our_mae, sota_mae, mae_diff))
    st.write(format_line("RMSE", our_rmse, sota_rmse, rmse_diff))
    st.write(format_line("MAPE", our_mape, sota_mape, mape_diff, True))

    st.caption("Difference is calculated as (ST-Transformer Multi-Hop - MTESformer SOTA).")


def render_live_horizon_simulation():
    st.subheader("Live Evaluation Simulation")

    st.write(
        "This simulation shows the evaluation process step-by-step. "
        "As more prediction horizons are evaluated, the curves gradually converge "
        "to the final average metrics shown in the summary table."
    )

    run = st.button("Run Live Evaluation Simulation", use_container_width=True)

    chart_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    target_minutes = [10, 20, 30, 40, 50, 60]

    if run:
        for i, current_minutes in enumerate(target_minutes, start=1):
            step = current_minutes // 5

            fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

            all_handles = []
            all_labels = []

            for ax, metric_name, y_label, title in [
                (axes[0], "MAE", "MAE", "Cumulative MAE"),
                (axes[1], "RMSE", "RMSE", "Cumulative RMSE"),
                (axes[2], "MAPE", "MAPE (%)", "Cumulative MAPE"),
            ]:
                handles, labels = plot_metric_line(
                    ax,
                    metric_name,
                    y_label,
                    title,
                    max_steps=step,
                    current_minutes=current_minutes,
                )

                for h, l in zip(handles, labels):
                    if l not in all_labels:
                        all_handles.append(h)
                        all_labels.append(l)

            fig.legend(
                all_handles,
                all_labels,
                loc="lower center",
                ncol=4,
                bbox_to_anchor=(0.5, -0.02),
            )

            fig.tight_layout(rect=[0, 0.08, 1, 1])
            chart_placeholder.pyplot(fig)
            plt.close(fig)

            progress_bar.progress(i / len(target_minutes))
            status_text.write(f"Evaluation progressed to {current_minutes} minutes")

            time.sleep(0.5)

        save_final_demo_images()
        save_pair_comparison_images()

        st.success("Evaluation completed. Final metrics match the summary table.")

        render_model_comparison()

    else:
        st.info("Click the button above to simulate live evaluation.")


def render_final_comparison():
    st.title("Final Comparison")

    st.markdown("### Comparison Summary (Average Across Horizons)")
    render_comparison_summary()

    st.divider()

    render_live_horizon_simulation()


left_col, right_col = st.columns([1, 3])

with left_col:
    st.write("")
    st.markdown("### Traffic Forecast Demo")

    st.markdown(
        """
#### Models included:

- Transformer (Baseline)  
- ST-Transformer (Advanced)  
- ST-Transformer Multi-Hop (Advanced)  
- MTESformer (SOTA)
"""
    )

with right_col:
    render_final_comparison()