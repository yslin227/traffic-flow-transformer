import os
import sys
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

TRANSFORMER_FILE = os.path.join(
    PROJECT_ROOT, "results", "transformer_results_comprehensive.txt"
)
ST_FILE = os.path.join(PROJECT_ROOT, "results", "results_metr_la_comprehensive.txt")
MULTIHOP_FILE = os.path.join(
    PROJECT_ROOT, "results", "results_multihop_comprehensive.txt"
)

st.set_page_config(
    page_title="Traffic Forecast Demo",
    page_icon="📈",
    layout="wide",
)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "active_section" not in st.session_state:
    st.session_state.active_section = "Model Description"


def select_model(model_name: str):
    st.session_state.selected_model = model_name
    st.session_state.active_section = "Model Description"


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
            elif line.startswith("Model Type:"):
                metrics["Model"] = line.split(":", 1)[1].strip()
            elif line.startswith("MAE:"):
                metrics["MAE"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("RMSE:"):
                metrics["RMSE"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("WMAPE:"):
                metrics["WMAPE (%)"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("R² Score:") or line.startswith("R2 Score:"):
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
    transformer_metrics = parse_metrics(TRANSFORMER_FILE)
    st_metrics = parse_metrics(ST_FILE)
    multihop_metrics = parse_metrics(MULTIHOP_FILE)

    return pd.DataFrame(
        {
            "Model": ["Transformer", "ST-Transformer", "ST-Transformer Multi-Hop"],
            "Type": ["Benchmark", "Advanced", "Advanced+"],
            "MAE": [
                transformer_metrics["MAE"],
                st_metrics["MAE"],
                multihop_metrics["MAE"],
            ],
            "RMSE": [
                transformer_metrics["RMSE"],
                st_metrics["RMSE"],
                multihop_metrics["RMSE"],
            ],
            "WMAPE (%)": [
                transformer_metrics["WMAPE (%)"],
                st_metrics["WMAPE (%)"],
                multihop_metrics["WMAPE (%)"],
            ],
            "R²": [
                transformer_metrics["R²"],
                st_metrics["R²"],
                multihop_metrics["R²"],
            ],
            "Best Validation Loss": [
                transformer_metrics["Best Validation Loss"],
                st_metrics["Best Validation Loss"],
                multihop_metrics["Best Validation Loss"],
            ],
        }
    )


def render_section_selector():
    options = ["Model Description", "Workflow", "Result Visualization"]

    section = st.radio(
        "Section",
        options,
        horizontal=True,
        index=options.index(st.session_state.active_section),
        label_visibility="collapsed",
        key="section_radio",
    )

    st.session_state.active_section = section
    return section


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
    section = render_section_selector()

    if section == "Model Description":
        st.markdown("### Model Description")
        st.markdown(
            """
            Transformer is used here as the **benchmark model**.
            It uses **self-attention** to model relationships across the full input sequence.
            Compared with recurrent models, this allows it to better capture longer-range
            temporal dependencies. However, this model mainly focuses on temporal modeling
            and does not explicitly use graph-based spatial relationships between sensors.
            """
        )
        st.write("")
        render_metric_table("Transformer Result Summary", metrics)

    elif section == "Workflow":
        render_transformer_workflow()

    elif section == "Result Visualization":
        render_result_plots("Transformer", horizon_df)


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
    section = render_section_selector()

    if section == "Model Description":
        st.markdown("### Model Description")
        st.markdown(
            """
            ST-Transformer is an **advanced spatio-temporal model** for traffic forecasting.
            Unlike the standard Transformer, it models both:

            - **Temporal dependencies** across time
            - **Spatial dependencies** across sensors

            This makes it more suitable for traffic forecasting, where nearby sensors and
            road-network relationships strongly influence future traffic states.
            """
        )
        st.write("")
        render_metric_table("ST-Transformer Result Summary", metrics)

    elif section == "Workflow":
        render_st_workflow()

    elif section == "Result Visualization":
        render_result_plots("ST-Transformer", horizon_df)


def render_multihop_workflow():
    st.subheader("Workflow")
    st.markdown(
        """
        **ST-Transformer Multi-Hop pipeline**

        1. Load METR-LA traffic data  
        2. Build temporal features  
        3. Create sliding windows (12 input steps → 12 prediction steps)  
        4. Construct multi-hop graph bias using A, A², and A³  
        5. Apply temporal attention  
        6. Apply spatial attention with multi-hop graph structure  
        7. Fuse spatial-temporal representations  
        8. Generate multi-step traffic predictions  
        9. Evaluate with MAE / RMSE / WMAPE / R²
        """
    )

    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#FDEDEC", color="#922B21"];
            A [label="METR-LA Data"];
            B [label="Time Features"];
            C [label="Sliding Window\\n12 -> 12"];
            D [label="Multi-Hop Graph\\nA, A², A³"];
            E [label="Temporal Attention"];
            F [label="Spatial Attention"];
            G [label="ST Fusion"];
            H [label="Prediction"];
            I [label="Evaluation"];

            A -> B -> C -> D;
            C -> E;
            D -> F;
            E -> G;
            F -> G;
            G -> H -> I;
        }
        """
    )


def render_multihop_page():
    metrics = parse_metrics(MULTIHOP_FILE)
    horizon_df = parse_horizon_metrics(MULTIHOP_FILE)

    st.title("ST-Transformer Multi-Hop Results")
    section = render_section_selector()

    if section == "Model Description":
        st.markdown("### Model Description")
        st.markdown(
            """
            ST-Transformer Multi-Hop extends the original ST-Transformer by using
            **multi-hop graph bias** in spatial attention.

            Instead of only considering direct 1-hop neighbors, the model incorporates:

            - **1-hop adjacency**: A
            - **2-hop adjacency**: A²
            - **3-hop adjacency**: A³

            These graph relationships are combined with learnable weights:

            **w1·A + w2·A² + w3·A³**

            This allows the model to capture longer-range spatial traffic interactions
            and improves forecasting performance across multiple prediction horizons.
            """
        )
        st.write("")
        render_metric_table("Multi-Hop Result Summary", metrics)

    elif section == "Workflow":
        render_multihop_workflow()

    elif section == "Result Visualization":
        render_result_plots("ST-Transformer Multi-Hop", horizon_df)


def render_comparison_summary():
    df = load_comparison_table()

    display_df = df.copy()
    for col in ["MAE", "RMSE", "WMAPE (%)", "R²", "Best Validation Loss"]:
        display_df[col] = display_df[col].map(
            lambda x: f"{x:.4f}" if pd.notnull(x) else "-"
        )

    st.table(display_df.set_index("Model"))


def plot_metric_line(ax, metric_name, y_label, title):
    data_sources = [
        ("Transformer", parse_horizon_metrics(TRANSFORMER_FILE), "-", "o"),
        ("ST-Transformer", parse_horizon_metrics(ST_FILE), "--", "s"),
        ("Multi-Hop", parse_horizon_metrics(MULTIHOP_FILE), ":", "^"),
    ]

    for label, df, linestyle, marker in data_sources:
        if not df.empty:
            ax.plot(
                df["Minutes"],
                df[metric_name],
                linestyle=linestyle,
                marker=marker,
                label=label,
            )

    ax.set_xlabel("Prediction Horizon (Minutes)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def render_horizon_comparison():
    col1, col2 = st.columns(2)

    with col1:
        fig_mae, ax_mae = plt.subplots(figsize=(6, 4))
        plot_metric_line(ax_mae, "MAE", "MAE", "MAE across Horizon")
        st.pyplot(fig_mae)

    with col2:
        fig_rmse, ax_rmse = plt.subplots(figsize=(6, 4))
        plot_metric_line(ax_rmse, "RMSE", "RMSE", "RMSE across Horizon")
        st.pyplot(fig_rmse)

    col3, col4 = st.columns(2)

    with col3:
        fig_wmape, ax_wmape = plt.subplots(figsize=(6, 4))
        plot_metric_line(ax_wmape, "WMAPE", "WMAPE (%)", "WMAPE across Horizon")
        st.pyplot(fig_wmape)

    with col4:
        fig_r2, ax_r2 = plt.subplots(figsize=(6, 4))
        plot_metric_line(ax_r2, "R2", "R²", "R² across Horizon")
        st.pyplot(fig_r2)


def render_comparison_page():
    st.title("Final Comparison")

    st.markdown("### Comparison Summary")
    render_comparison_summary()

    st.divider()

    st.markdown("### Horizon Comparison")
    render_horizon_comparison()


left_col, right_col = st.columns([1, 3])

with left_col:
    st.markdown("## Traffic Forecast Demo")
    st.info("Model Selection")

    if st.button("Transformer (Benchmark)", use_container_width=True):
        select_model("Transformer")

    if st.button("ST-Transformer (Advanced)", use_container_width=True):
        select_model("ST-Transformer")

    if st.button("ST-Transformer Multi-Hop", use_container_width=True):
        select_model("Multi-Hop")

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

with right_col:
    if st.session_state.selected_model is None:
        st.markdown("## Select an option from the left panel")

    elif st.session_state.selected_model == "Transformer":
        render_transformer_page()

    elif st.session_state.selected_model == "ST-Transformer":
        render_st_page()

    elif st.session_state.selected_model == "Multi-Hop":
        render_multihop_page()

    elif st.session_state.selected_model == "Comparison":
        render_comparison_page()