import os
import sys
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# Path Setup
# ======================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

# ======================
# File Paths
# ======================
LSTM_FILE = os.path.join(PROJECT_ROOT, "results", "lstm_results_comprehensive.txt")
TRANSFORMER_FILE = os.path.join(
    PROJECT_ROOT, "results", "transformer_results_comprehensive.txt"
)
ST_FILE = os.path.join(PROJECT_ROOT, "results", "results_metr_la_comprehensive.txt")

# ======================
# Page Config
# ======================
st.set_page_config(
    page_title="Traffic Forecast Demo",
    page_icon="📈",
    layout="wide",
)

# ======================
# Session State
# ======================
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None


def select_model(model_name: str):
    st.session_state.selected_model = model_name


# ======================
# Helpers
# ======================
def safe_float(value: str):
    try:
        match = re.search(r"-?\d+(\.\d+)?", value)
        return float(match.group()) if match else None
    except (ValueError, AttributeError):
        return None


def parse_metrics(file_path: str):
    metrics = {
        "Model": None,
        "MAE": None,
        "RMSE": None,
        "WMAPE (%)": None,
        "R²": None,
        "Best Validation Loss": None,
    }

    if not os.path.exists(file_path):
        return metrics

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line.startswith("Model:"):
                metrics["Model"] = line.split(":", 1)[1].strip()
            elif line.startswith("MAE:"):
                metrics["MAE"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("RMSE:"):
                metrics["RMSE"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("WMAPE:"):
                metrics["WMAPE (%)"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("R² Score:"):
                metrics["R²"] = safe_float(line.split(":", 1)[1])
            elif "Best Validation Loss" in line:
                metrics["Best Validation Loss"] = safe_float(line.split(":", 1)[1])
            elif "Best Validation MAE" in line:
                metrics["Best Validation Loss"] = safe_float(line.split(":", 1)[1])

    return metrics


def parse_horizon_metrics(file_path: str):
    rows = []

    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["Minutes", "MAE", "RMSE", "WMAPE", "R2"])

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
                        rows.append(
                            {
                                "Minutes": int(parts[1]),
                                "MAE": float(parts[2]),
                                "RMSE": float(parts[3]),
                                "WMAPE": safe_float(parts[4]),
                                "R2": float(parts[5]),
                            }
                        )
                    except ValueError:
                        continue

    return pd.DataFrame(rows)


def load_comparison_table():
    lstm_metrics = parse_metrics(LSTM_FILE)
    transformer_metrics = parse_metrics(TRANSFORMER_FILE)
    st_metrics = parse_metrics(ST_FILE)

    return pd.DataFrame(
        {
            "Model": ["LSTM", "Transformer", "ST-Transformer"],
            "Type": ["Benchmark", "Improved Benchmark", "Advanced"],
            "MAE": [
                lstm_metrics["MAE"],
                transformer_metrics["MAE"],
                st_metrics["MAE"],
            ],
            "RMSE": [
                lstm_metrics["RMSE"],
                transformer_metrics["RMSE"],
                st_metrics["RMSE"],
            ],
            "WMAPE (%)": [
                lstm_metrics["WMAPE (%)"],
                transformer_metrics["WMAPE (%)"],
                st_metrics["WMAPE (%)"],
            ],
            "R²": [
                lstm_metrics["R²"],
                transformer_metrics["R²"],
                st_metrics["R²"],
            ],
            "Best Validation Loss": [
                lstm_metrics["Best Validation Loss"],
                transformer_metrics["Best Validation Loss"],
                st_metrics["Best Validation Loss"],
            ],
        }
    )


def render_section_selector(model_key: str):
    return st.radio(
        "Section",
        ["Model Description", "Workflow", "Result Visualization"],
        horizontal=True,
        index=0,
        key=f"section_{model_key}",
        label_visibility="collapsed",
    )


# ======================
# Shared UI Components
# ======================
def render_metric_table(title: str, metrics: dict):
    st.subheader(title)

    table_df = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE",
                "WMAPE (%)",
                "R²",
                "Best Validation Loss",
            ],
            "Value": [
                f"{metrics['MAE']:.4f}" if metrics["MAE"] is not None else "-",
                f"{metrics['RMSE']:.4f}" if metrics["RMSE"] is not None else "-",
                f"{metrics['WMAPE (%)']:.2f}" if metrics["WMAPE (%)"] is not None else "-",
                f"{metrics['R²']:.4f}" if metrics["R²"] is not None else "-",
                f"{metrics['Best Validation Loss']:.4f}"
                if metrics["Best Validation Loss"] is not None
                else "-",
            ],
        }
    )

    st.table(table_df.set_index("Metric"))


def render_result_plots(model_name: str, horizon_df: pd.DataFrame):
    st.subheader("Result Visualization")

    if horizon_df.empty:
        st.warning(f"No horizon data found for {model_name}.")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig_mae, ax_mae = plt.subplots(figsize=(6, 4))
        ax_mae.plot(horizon_df["Minutes"], horizon_df["MAE"], marker="o")
        ax_mae.set_xlabel("Prediction Horizon (Minutes)")
        ax_mae.set_ylabel("MAE")
        ax_mae.set_title(f"{model_name} Per-Horizon MAE")
        ax_mae.grid(True)
        st.pyplot(fig_mae)

    with col2:
        fig_rmse, ax_rmse = plt.subplots(figsize=(6, 4))
        ax_rmse.plot(horizon_df["Minutes"], horizon_df["RMSE"], marker="o")
        ax_rmse.set_xlabel("Prediction Horizon (Minutes)")
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.set_title(f"{model_name} Per-Horizon RMSE")
        ax_rmse.grid(True)
        st.pyplot(fig_rmse)

    col3, col4 = st.columns(2)

    with col3:
        fig_combined, ax_combined = plt.subplots(figsize=(6, 4))
        ax_combined.plot(horizon_df["Minutes"], horizon_df["MAE"], marker="o", label="MAE")
        ax_combined.plot(horizon_df["Minutes"], horizon_df["RMSE"], marker="s", label="RMSE")
        ax_combined.set_xlabel("Prediction Horizon (Minutes)")
        ax_combined.set_ylabel("Error")
        ax_combined.set_title(f"{model_name} Error Comparison")
        ax_combined.legend()
        ax_combined.grid(True)
        st.pyplot(fig_combined)

    with col4:
        fig_r2, ax_r2 = plt.subplots(figsize=(6, 4))
        ax_r2.plot(horizon_df["Minutes"], horizon_df["R2"], marker="o")
        ax_r2.set_xlabel("Prediction Horizon (Minutes)")
        ax_r2.set_ylabel("R²")
        ax_r2.set_title(f"{model_name} R² across Horizon")
        ax_r2.grid(True)
        st.pyplot(fig_r2)


# ======================
# LSTM Components
# ======================
def render_lstm_workflow():
    st.subheader("Workflow")
    st.markdown(
        """
        **LSTM pipeline**

        1. Load METR-LA traffic data  
        2. Build time features (`time_of_day`, `day_of_week`)  
        3. Create sliding windows (12 input steps → 12 prediction steps)  
        4. Normalize input sequences  
        5. Feed sequences into LSTM  
        6. Predict future traffic values for all sensors  
        7. Evaluate with MAE / RMSE / WMAPE / R²
        """
    )

    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#5D6D7E"];
            A [label="METR-LA Data"];
            B [label="Time Features"];
            C [label="Sliding Window\\n12 -> 12"];
            D [label="Normalization"];
            E [label="LSTM Model"];
            F [label="Prediction"];
            G [label="Evaluation"];

            A -> B -> C -> D -> E -> F -> G;
        }
        """
    )


def render_lstm_page():
    metrics = parse_metrics(LSTM_FILE)
    horizon_df = parse_horizon_metrics(LSTM_FILE)

    st.title("LSTM Results")
    section = render_section_selector("LSTM")

    if section == "Model Description":
        st.markdown("### Model Description")
        st.markdown(
            """
            LSTM is used here as the **benchmark baseline** for traffic forecasting.
            It processes input sequences step by step through recurrent hidden states,
            which makes it effective for capturing short-term temporal dependencies.
            However, it mainly focuses on **temporal patterns** and does not explicitly
            model **spatial relationships** between traffic sensors.
            """
        )
        st.write("")
        render_metric_table("LSTM Result Summary", metrics)

    elif section == "Workflow":
        render_lstm_workflow()

    elif section == "Result Visualization":
        render_result_plots("LSTM", horizon_df)


# ======================
# Transformer Components
# ======================
def render_transformer_workflow():
    st.subheader("Workflow")
    st.markdown(
        """
        **Transformer pipeline**

        1. Load METR-LA traffic data  
        2. Build time features (`time_of_day`, `day_of_week`)  
        3. Create sliding windows (12 input steps → 12 prediction steps)  
        4. Normalize input sequences  
        5. Apply input projection + positional encoding  
        6. Use self-attention to capture global temporal dependencies  
        7. Generate multi-step traffic predictions  
        8. Evaluate with MAE / RMSE / WMAPE / R²
        """
    )

    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#E8F8F5", color="#117A65"];
            A [label="METR-LA Data"];
            B [label="Time Features"];
            C [label="Sliding Window\\n12 -> 12"];
            D [label="Normalization"];
            E [label="Input Projection"];
            F [label="Positional Encoding"];
            G [label="Transformer Encoder"];
            H [label="Forecast Head"];
            I [label="Prediction"];
            J [label="Evaluation"];

            A -> B -> C -> D -> E -> F -> G -> H -> I -> J;
        }
        """
    )


def render_transformer_page():
    metrics = parse_metrics(TRANSFORMER_FILE)
    horizon_df = parse_horizon_metrics(TRANSFORMER_FILE)

    st.title("Transformer Results")
    section = render_section_selector("Transformer")

    if section == "Model Description":
        st.markdown("### Model Description")
        st.markdown(
            """
            Transformer is used here as an **improved benchmark** over LSTM.
            Instead of processing sequences step by step, it uses **self-attention**
            to model the relationships across the entire input sequence at once.
            This makes it more effective for capturing **long-range temporal dependencies**.
            However, it still mainly focuses on **temporal modeling** and does not explicitly
            incorporate the spatial relationships between traffic sensors.
            """
        )
        st.write("")
        render_metric_table("Transformer Result Summary", metrics)

    elif section == "Workflow":
        render_transformer_workflow()

    elif section == "Result Visualization":
        render_result_plots("Transformer", horizon_df)


# ======================
# ST-Transformer Components
# ======================
def render_st_workflow():
    st.subheader("Workflow")
    st.markdown(
        """
        **ST-Transformer pipeline**

        1. Load METR-LA traffic data  
        2. Build time features and organize traffic sensor sequences  
        3. Create sliding windows (12 input steps → 12 prediction steps)  
        4. Encode temporal information through attention  
        5. Encode spatial relationships across sensors  
        6. Fuse spatial and temporal representations  
        7. Generate multi-step predictions for all sensors  
        8. Evaluate with MAE / RMSE / WMAPE / R²
        """
    )

    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#FEF5E7", color="#AF601A"];
            A [label="METR-LA Data"];
            B [label="Time Features"];
            C [label="Sliding Window\\n12 -> 12"];
            D [label="Temporal Attention"];
            E [label="Spatial Attention"];
            F [label="ST Fusion"];
            G [label="Prediction"];
            H [label="Evaluation"];

            A -> B -> C -> D;
            C -> E;
            D -> F;
            E -> F;
            F -> G -> H;
        }
        """
    )


def render_st_page():
    metrics = parse_metrics(ST_FILE)
    horizon_df = parse_horizon_metrics(ST_FILE)

    st.title("ST-Transformer Results")
    section = render_section_selector("ST-Transformer")

    if section == "Model Description":
        st.markdown("### Model Description")
        st.markdown(
            """
            ST-Transformer is the **advanced model** in this demo.
            It is specifically designed for **traffic forecasting**, which is fundamentally
            a **spatio-temporal problem**. Unlike LSTM and the standard Transformer,
            ST-Transformer models both:

            - **Temporal dependencies** across time
            - **Spatial dependencies** across traffic sensors

            This allows it to better capture complex traffic dynamics and generally
            achieve the strongest performance among the three models.
            """
        )
        st.write("")
        render_metric_table("ST-Transformer Result Summary", metrics)

    elif section == "Workflow":
        render_st_workflow()

    elif section == "Result Visualization":
        render_result_plots("ST-Transformer", horizon_df)


# ======================
# Final Comparison Page
# ======================
def render_comparison_summary():
    df = load_comparison_table()

    display_df = df.copy()
    for col in ["MAE", "RMSE", "WMAPE (%)", "R²", "Best Validation Loss"]:
        display_df[col] = display_df[col].map(
            lambda x: f"{x:.4f}" if pd.notnull(x) else "-"
        )

    st.table(display_df.set_index("Model"))


def render_horizon_comparison():
    st.subheader("Per-Horizon Comparison")

    lstm_df = parse_horizon_metrics(LSTM_FILE)
    transformer_df = parse_horizon_metrics(TRANSFORMER_FILE)
    st_df = parse_horizon_metrics(ST_FILE)

    line_styles = {
        "LSTM": "-",
        "Transformer": "--",
        "ST-Transformer": ":",
    }
    markers = {
        "LSTM": "o",
        "Transformer": "s",
        "ST-Transformer": "^",
    }

    col1, col2 = st.columns(2)

    with col1:
        fig_mae, ax_mae = plt.subplots(figsize=(6, 4))
        if not lstm_df.empty:
            ax_mae.plot(
                lstm_df["Minutes"], lstm_df["MAE"],
                linestyle=line_styles["LSTM"], marker=markers["LSTM"], label="LSTM"
            )
        if not transformer_df.empty:
            ax_mae.plot(
                transformer_df["Minutes"], transformer_df["MAE"],
                linestyle=line_styles["Transformer"], marker=markers["Transformer"], label="Transformer"
            )
        if not st_df.empty:
            ax_mae.plot(
                st_df["Minutes"], st_df["MAE"],
                linestyle=line_styles["ST-Transformer"], marker=markers["ST-Transformer"], label="ST-Transformer"
            )
        ax_mae.set_xlabel("Prediction Horizon (Minutes)")
        ax_mae.set_ylabel("MAE")
        ax_mae.set_title("MAE across Horizon")
        ax_mae.legend()
        ax_mae.grid(True)
        st.pyplot(fig_mae)

    with col2:
        fig_rmse, ax_rmse = plt.subplots(figsize=(6, 4))
        if not lstm_df.empty:
            ax_rmse.plot(
                lstm_df["Minutes"], lstm_df["RMSE"],
                linestyle=line_styles["LSTM"], marker=markers["LSTM"], label="LSTM"
            )
        if not transformer_df.empty:
            ax_rmse.plot(
                transformer_df["Minutes"], transformer_df["RMSE"],
                linestyle=line_styles["Transformer"], marker=markers["Transformer"], label="Transformer"
            )
        if not st_df.empty:
            ax_rmse.plot(
                st_df["Minutes"], st_df["RMSE"],
                linestyle=line_styles["ST-Transformer"], marker=markers["ST-Transformer"], label="ST-Transformer"
            )
        ax_rmse.set_xlabel("Prediction Horizon (Minutes)")
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.set_title("RMSE across Horizon")
        ax_rmse.legend()
        ax_rmse.grid(True)
        st.pyplot(fig_rmse)

    col3, col4 = st.columns(2)

    with col3:
        fig_wmape, ax_wmape = plt.subplots(figsize=(6, 4))
        if not lstm_df.empty:
            ax_wmape.plot(
                lstm_df["Minutes"], lstm_df["WMAPE"],
                linestyle=line_styles["LSTM"], marker=markers["LSTM"], label="LSTM"
            )
        if not transformer_df.empty:
            ax_wmape.plot(
                transformer_df["Minutes"], transformer_df["WMAPE"],
                linestyle=line_styles["Transformer"], marker=markers["Transformer"], label="Transformer"
            )
        if not st_df.empty:
            ax_wmape.plot(
                st_df["Minutes"], st_df["WMAPE"],
                linestyle=line_styles["ST-Transformer"], marker=markers["ST-Transformer"], label="ST-Transformer"
            )
        ax_wmape.set_xlabel("Prediction Horizon (Minutes)")
        ax_wmape.set_ylabel("WMAPE (%)")
        ax_wmape.set_title("WMAPE across Horizon")
        ax_wmape.legend()
        ax_wmape.grid(True)
        st.pyplot(fig_wmape)

    with col4:
        fig_r2, ax_r2 = plt.subplots(figsize=(6, 4))
        if not lstm_df.empty:
            ax_r2.plot(
                lstm_df["Minutes"], lstm_df["R2"],
                linestyle=line_styles["LSTM"], marker=markers["LSTM"], label="LSTM"
            )
        if not transformer_df.empty:
            ax_r2.plot(
                transformer_df["Minutes"], transformer_df["R2"],
                linestyle=line_styles["Transformer"], marker=markers["Transformer"], label="Transformer"
            )
        if not st_df.empty:
            ax_r2.plot(
                st_df["Minutes"], st_df["R2"],
                linestyle=line_styles["ST-Transformer"], marker=markers["ST-Transformer"], label="ST-Transformer"
            )
        ax_r2.set_xlabel("Prediction Horizon (Minutes)")
        ax_r2.set_ylabel("R²")
        ax_r2.set_title("R² across Horizon")
        ax_r2.legend()
        ax_r2.grid(True)
        st.pyplot(fig_r2)


def render_comparison_page():
    st.title("Final Comparison")

    st.markdown("### Comparison Summary")
    render_comparison_summary()

    st.divider()

    st.markdown("### Horizon Comparison")
    render_horizon_comparison()


# ======================
# Layout
# ======================
left_col, right_col = st.columns([1, 3])

# ======================
# Left Panel
# ======================
with left_col:
    st.markdown("## Traffic Forecast Demo")
    st.info("Model Selection")

    if st.button("LSTM (Benchmark)", use_container_width=True):
        select_model("LSTM")

    if st.button("Transformer (Improved Benchmark)", use_container_width=True):
        select_model("Transformer")

    if st.button("ST-Transformer (Advanced)", use_container_width=True):
        select_model("ST-Transformer")

    st.divider()

    if st.button("📊 Final Comparison", use_container_width=True):
        select_model("Comparison")

    st.divider()

    st.write("### Current View")
    if st.session_state.selected_model == "Comparison":
        st.success("Final Comparison")
    elif st.session_state.selected_model:
        st.success(st.session_state.selected_model)
    else:
        st.warning("No selection")

# ======================
# Right Panel
# ======================
with right_col:
    if st.session_state.selected_model is None:
        st.markdown("## Select an option from the left panel")

    elif st.session_state.selected_model == "LSTM":
        render_lstm_page()

    elif st.session_state.selected_model == "Transformer":
        render_transformer_page()

    elif st.session_state.selected_model == "ST-Transformer":
        render_st_page()

    elif st.session_state.selected_model == "Comparison":
        render_comparison_page()