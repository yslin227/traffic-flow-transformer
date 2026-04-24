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

st.set_page_config(page_title="Traffic Forecast Demo", layout="wide")

PAPER_RESULTS = {
    "Model": "MTESformer (Paper)",
    "MAE": 3.37,
    "RMSE": 7.14,
    "MAPE (%)": 9.62,
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
            elif line.startswith("MAPE:") or line.startswith("MAPE (>0.1):"):
                if metrics["MAPE (%)"] is None:
                    metrics["MAPE (%)"] = safe_float(line.split(":", 1)[1])

    return metrics


def parse_horizon_metrics(file_path: str):
    rows = []

    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["Minutes", "MAE", "RMSE"])

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
                if "SENSOR-LEVEL STATISTICS" in stripped:
                    break

                parts = stripped.split()
                if len(parts) >= 6:
                    try:
                        rows.append({
                            "Minutes": int(parts[1]),
                            "MAE": float(parts[2]),
                            "RMSE": float(parts[3]),
                        })
                    except ValueError:
                        continue

    return pd.DataFrame(rows)


def load_comparison_table():
    transformer = parse_metrics(TRANSFORMER_FILE)
    st_model = parse_metrics(ST_FILE)
    multihop = parse_metrics(MULTIHOP_FILE)

    return pd.DataFrame({
        "Model": [
            "Transformer (Benchmark)",
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
    })


def render_comparison_summary():
    st.caption("The comparison summary shows average performance across prediction horizons.")

    df = load_comparison_table()
    display_df = df.copy()

    for col in ["MAE", "RMSE", "MAPE (%)"]:
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
            "MTESformer (Paper)",
            PAPER_RESULTS["Horizon"],
            {
                "MAE": PAPER_RESULTS["MAE"],
                "RMSE": PAPER_RESULTS["RMSE"],
                "MAPE (%)": PAPER_RESULTS["MAPE (%)"],
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

    if metric_name == "MAPE":
        if "MAPE" in plot_df.columns:
            y_values = plot_df["MAPE"].expanding().mean()
        else:
            overall_mape = metrics.get("MAPE (%)")
            if overall_mape is None:
                return None
            y_values = pd.Series([overall_mape] * len(plot_df))
    else:
        if metric_name not in plot_df.columns:
            return None
        y_values = plot_df[metric_name].expanding().mean()

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

        if label == "MTESformer (Paper)" and current_minutes is not None:
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
        )

        if len(plot_df) > 0:
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


def render_model_comparison():
    multihop = parse_metrics(MULTIHOP_FILE)

    our_mae = multihop["MAE"]
    our_rmse = multihop["RMSE"]
    our_mape = multihop["MAPE (%)"]

    paper_mae = PAPER_RESULTS["MAE"]
    paper_rmse = PAPER_RESULTS["RMSE"]
    paper_mape = PAPER_RESULTS["MAPE (%)"]

    def calc_diff(our, paper):
        if our is None or paper is None:
            return None
        return our - paper

    mae_diff = calc_diff(our_mae, paper_mae)
    rmse_diff = calc_diff(our_rmse, paper_rmse)
    mape_diff = calc_diff(our_mape, paper_mape)

    st.divider()
    st.subheader("Final Model Comparison")

    st.markdown("**ST-Transformer Multi-Hop (Advanced) vs MTESformer (Paper)**")

    def format_line(name, our, paper, diff, is_percent=False):
        if our is None or paper is None:
            return f"{name}: -"

        sign = "+" if diff > 0 else ""

        if is_percent:
            return f"{name}: {our:.2f}% vs {paper:.2f}%  ({sign}{diff:.2f}%)"
        return f"{name}: {our:.2f} vs {paper:.2f}  ({sign}{diff:.2f})"

    st.write(format_line("MAE", our_mae, paper_mae, mae_diff))
    st.write(format_line("RMSE", our_rmse, paper_rmse, rmse_diff))
    st.write(format_line("MAPE", our_mape, paper_mape, mape_diff, True))

    st.caption("Difference is calculated as (ST-Transformer Multi-Hop - MTESformer (Paper)).")


def render_live_horizon_simulation():
    st.subheader("Live Evaluation Simulation")

    st.write(
        "This simulation shows the evaluation process step-by-step. "
        "As more prediction horizons are evaluated, the curves gradually converge "
        "to the final average metrics shown in the summary table."
    )

    st.caption(
        "Each point shows cumulative average performance up to that horizon. "
        "The final point matches the overall test performance."
    )

    run = st.button("Run Live Evaluation Simulation", use_container_width=True)

    chart_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    target_minutes = [10, 20, 30, 40, 50, 60]

    if run:
        for i, current_minutes in enumerate(target_minutes, start=1):

            step = current_minutes // 5  # 對應 index

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

            status_text.write(
                f"Evaluation progressed to {current_minutes} minutes"
            )

            time.sleep(0.5)

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

- Transformer (Benchmark)  
- ST-Transformer (Advanced)  
- ST-Transformer Multi-Hop (Advanced)  
- MTESformer (Paper)
"""
    )

with right_col:
    render_final_comparison()