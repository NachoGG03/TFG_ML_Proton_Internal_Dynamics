import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from IPython.display import display


def pid_to_latex(pid):
    """
    Translate the numbers of PIDs into LaTeX text
    """
    translate = {21: "g", 1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t"}
    flav = translate[abs(pid)]
    if pid < 0:
        flav = rf"\bar{{{flav}}}"
    return f"${flav}$"


def plot_LHAPDF(input_xgrid, input_q2grid, val_pdf, pids, scales, flavour_select):
    """
    Generates and displays PDF plots for different Q scales.
    Manual saving can be done from the plot window.
    """
    unique_q2 = np.unique(input_q2grid)

    for q_gev in scales:
        target_q2 = q_gev**2
        # Find the closest Q2 value in the grid
        q2_val = unique_q2[np.argmin(np.abs(unique_q2 - target_q2))]

        mask = input_q2grid == q2_val
        x_plot = input_xgrid[mask]
        pdf_plot = val_pdf[mask]
        pdf_grid = defaultdict(list)

        # Populate the flavor dictionary
        for point in pdf_plot:
            for pid in flavour_select:
                # Assuming pid_to_latex is defined globally
                label = pid_to_latex(pid)
                pdf_grid[label].append(point[pids.index(pid)])

        # Plot configuration
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"PDF plots for NNPDF40 set at Q={q_gev} GeV", fontsize=16)

        for i, (parton, val) in enumerate(pdf_grid.items()):
            ax = axes.flatten()[i]
            ax.plot(x_plot, val, label=rf"{parton} LHAPDF")
            ax.set_xscale("log")
            ax.set_ylabel(rf"x {parton}(x)")
            if i >= 2:
                ax.set_xlabel("x")
            ax.legend()
            ax.grid(True, which="both", ls=":", alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def generate_physics_report(x, y_true, y_pred, f_map, q2_value, export_path=None):
    """
    Calculates sum rules, computes errors/fidelity, and displays a styled report.
    Optionally exports the table to a file.
    """

    if len(x) == 0:
        print(f"Error: Empty data for Q2 = {q2_value}.")
        return None

    # 1. --- CALCULATIONS (Sum Rules) ---
    safe_x = np.where(x == 0, 1e-10, x)

    # Momentum Rule: Integral of sum[x*f(x)] dx
    mom_t = np.trapz(np.sum([y_true[:, i] for i in f_map.values()], axis=0), x)
    mom_p = np.trapz(np.sum([y_pred[:, i] for i in f_map.values()], axis=0), x)

    # Valence Rules: Integral of [q(x) - q_bar(x)] dx = (xf - xf_bar)/x dx
    def get_valence(flavor):
        t = np.trapz(
            (y_true[:, f_map[flavor]] - y_true[:, f_map[f"{flavor}_bar"]]) / safe_x, x
        )
        p = np.trapz(
            (y_pred[:, f_map[flavor]] - y_pred[:, f_map[f"{flavor}_bar"]]) / safe_x, x
        )
        return t, p

    ut, up = get_valence("u")
    dt, dp = get_valence("d")
    st, sp = get_valence("s")
    ct, cp = get_valence("c")

    # 2. --- DATAFRAME CONSTRUCTION ---
    df = pd.DataFrame(
        {
            "Sum Rule": [
                "Total Momentum",
                "u Valence",
                "d Valence",
                "s Valence",
                "c Valence",
            ],
            "Theoretical": [1.0, 2.0, 1.0, 0.0, 0.0],
            "LHAPDF (Target)": [mom_t, ut, dt, st, ct],
            "NN (Prediction)": [mom_p, up, dp, sp, cp],
        }
    )

    # Metrics
    df["Abs Error"] = np.abs(df["LHAPDF (Target)"] - df["NN (Prediction)"])
    df["Fidelity (%)"] = (
        1 - (df["Abs Error"] / np.clip(np.abs(df["LHAPDF (Target)"]), 1e-4, None))
    ) * 100

    # 3. --- EXPORT (Optional) ---
    if export_path:
        if export_path.endswith(".tex"):
            df.to_latex(export_path, index=False, float_format="%.4f")
        else:
            df.to_csv(export_path, index=False)
        print(f"Report exported to: {export_path}")

    # 4. --- STYLING AND DISPLAY ---
    print(f"\n--- Physical Consistency Report (Q = {np.sqrt(q2_value):.2f} GeV) ---")

    styled_df = df.style.format(
        {
            "Theoretical": "{:.1f}",
            "LHAPDF (Target)": "{:.4f}",
            "NN (Prediction)": "{:.4f}",
            "Abs Error": "{:.2e}",
            "Fidelity (%)": "{:.2f}%",
        }
    ).background_gradient(subset=["Fidelity (%)"], cmap="RdYlGn", vmin=90, vmax=100)

    display(styled_df)
    return df  # Returns raw DF for further use


def plot_pdf_comparison(
    input_xgrid,
    input_q2grid,
    output_data,
    predictions,
    pids_to_plot,
    output_basis,
    pid_names,
    q_targets,
):
    """
    Generates a comparison grid of PDF plots (Main vs Prediction) and their ratios.

    Args:
        input_xgrid, input_q2grid: Input grids from the dataset.
        output_data: Ground truth (LHAPDF).
        predictions: Model predictions.
        pids_to_plot: List of PIDs to visualize.
        output_basis: Full list of PIDs in the model's output.
        pid_names: Human-readable names for the PIDs.
        selected_q2: List of Q2 values to plot (columns).
    """
    q2_values = [q**2 for q in q_targets]
    selected_q2 = []
    available_q2 = np.unique(input_q2grid)
    # 1. Find the closest Q2 values available in the grid
    for q2_target in q2_values:
        closest_val = available_q2[np.abs(available_q2 - q2_target).argmin()]
        selected_q2.append(closest_val)

    n_pids = len(pids_to_plot)
    n_q2 = len(selected_q2)

    # 2. Setup subplots: 2 rows per PID (Main + Ratio)
    fig, axes = plt.subplots(
        nrows=n_pids * 2,
        ncols=n_q2,
        figsize=(18, 4 * n_pids),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1] * n_pids},
    )

    # Handle cases where there might be only 1 PID or 1 Q2 value
    if n_pids == 1:
        axes = axes.reshape(2, n_q2)
    if n_q2 == 1:
        axes = axes.reshape(n_pids * 2, 1)

    for row_idx, pid in enumerate(pids_to_plot):
        idx_in_basis = output_basis.index(pid)
        main_row = row_idx * 2
        ratio_row = main_row + 1

        for col, q2_val in enumerate(selected_q2):
            # Masking and sorting data
            q2_mask = input_q2grid == q2_val
            x_plot = input_xgrid[q2_mask]
            real_plot = output_data[q2_mask, idx_in_basis]
            pred_plot = predictions[q2_mask, idx_in_basis]

            sort_idx = np.argsort(x_plot)
            x_plot = x_plot[sort_idx]
            real_plot = real_plot[sort_idx]
            pred_plot = pred_plot[sort_idx]

            # --- Main Plot ---
            ax_main = axes[main_row, col]
            ax_main.plot(x_plot, real_plot, "k-", lw=1.5, label="LHAPDF")
            ax_main.plot(x_plot, pred_plot, "r--", lw=1.5, label="NN Model")
            ax_main.set_xscale("log")
            ax_main.grid(True, which="both", ls=":", alpha=0.5)

            if row_idx == 0:
                # Showing the actual square root (Q) in the title for clarity
                ax_main.set_title(f"$Q = {np.sqrt(q2_val):.2f}$ GeV", fontsize=14)
            if col == 0:
                ax_main.set_ylabel(
                    f"{pid_names[row_idx]}\n$xf(x, Q^2)$", fontweight="bold"
                )
            if row_idx == 0 and col == n_q2 - 1:
                ax_main.legend()

            # --- Ratio Plot ---
            ax_ratio = axes[ratio_row, col]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = pred_plot / real_plot  # Typical convention: Prediction / Data

            ax_ratio.plot(x_plot, ratio, "blue", lw=1.2)
            ax_ratio.axhline(1.0, color="black", linestyle="-", alpha=0.7)
            ax_ratio.set_ylim(0.99, 1.01)
            ax_ratio.set_xscale("log")
            ax_ratio.grid(True, which="both", ls=":", alpha=0.5)

            if col == 0:
                ax_ratio.set_ylabel("NN/Data", fontsize=9)
            if row_idx == n_pids - 1:
                ax_ratio.set_xlabel("$x$", fontsize=12)

    plt.suptitle(
        "Model Comparative: NN vs LHAPDF with Ratio Analysis", fontsize=22, y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.subplots_adjust(hspace=0.15)

    return fig


def plot_pdf_valence(
    input_xgrid, input_q2grid, output_data, predictions, q_values_to_plot
):
    """
    Generates a comparison of valence distributions, gluons (scaled by 10),
    and sea flavors for different Q scales.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    q2_values_to_plot = [q**2 for q in q_values_to_plot]
    # 1. Figure configuration
    n_plots = len(q2_values_to_plot)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Index definition based on output_basis
    # Ensure these match the positions in your output array
    idx = {"u_bar": 2, "d_bar": 3, "g": 4, "d": 5, "u": 6, "s": 7, "c": 8}

    available_q2 = np.unique(input_q2grid)

    for i, q2_target in enumerate(q2_values_to_plot):
        if i >= len(axes):
            break  # Prevent error if more than 4 Q2 values are requested

        ax = axes[i]

        # Find the closest available Q2 in the dataset
        closest_q2 = available_q2[np.abs(available_q2 - q2_target).argmin()]
        closest_q = np.sqrt(closest_q2)

        # Masking and filtering
        mask_q2 = input_q2grid == closest_q2
        x_plot = input_xgrid[mask_q2]
        y_true = output_data[mask_q2]
        y_pred = predictions[mask_q2]

        # Sort by x to ensure continuous lines (prevent zig-zags)
        sort_idx = np.argsort(x_plot)
        x_s = x_plot[sort_idx]
        y_t_s = y_true[sort_idx]
        y_p_s = y_pred[sort_idx]

        # --- Physical Combination Calculations ---
        # Valence quarks
        uv_t, uv_p = (
            (y_t_s[:, idx["u"]] - y_t_s[:, idx["u_bar"]]),
            (y_p_s[:, idx["u"]] - y_p_s[:, idx["u_bar"]]),
        )
        dv_t, dv_p = (
            (y_t_s[:, idx["d"]] - y_t_s[:, idx["d_bar"]]),
            (y_p_s[:, idx["d"]] - y_p_s[:, idx["d_bar"]]),
        )

        # Gluon scaled for visibility
        g10_t, g10_p = y_t_s[:, idx["g"]] / 10, y_p_s[:, idx["g"]] / 10

        # Sea flavors and combinations
        flavors = [
            (uv_t, uv_p, r"$u_v$", "#0004ff"),
            (dv_t, dv_p, r"$d_v$", "#ffa600"),
            (g10_t, g10_p, r"$g/10$", "#ff0000"),
            (y_t_s[:, idx["u_bar"]], y_p_s[:, idx["u_bar"]], r"$\bar{u}$", "#c300ff"),
            (y_t_s[:, idx["d_bar"]], y_p_s[:, idx["d_bar"]], r"$\bar{d}$", "#0C6600"),
            (y_t_s[:, idx["s"]], y_p_s[:, idx["s"]], r"$s$", "#00fff2"),
            (y_t_s[:, idx["c"]], y_p_s[:, idx["c"]], r"$c$", "#00ff00"),
        ]

        # --- Line Rendering ---
        for true_val, pred_val, label, color in flavors:
            ax.plot(
                x_s, true_val, label=f"{label} (LHAPDF)", color=color, lw=1.2, alpha=0.6
            )
            ax.plot(x_s, pred_val, color=color, linestyle="--", lw=1.5)

        # Axis aesthetics
        ax.set_xscale("log")
        ax.set_xlim([1e-3, 1])
        ax.set_ylim([0, 1])
        ax.set_title(rf"PDFs at $Q = {closest_q:.2f}$ GeV", fontsize=14)
        ax.set_xlabel(r"$x$", fontsize=12)
        ax.set_ylabel(r"$xf(x, Q^2)$", fontsize=12)
        ax.grid(True, which="both", ls=":", alpha=0.5)

        if i == 0:
            ax.legend(loc="upper right", fontsize="x-small", ncol=2)

    plt.tight_layout()
    return fig


def plot_pdf_comparison_multi(
    input_xgrid,
    input_q2grid,
    output_data,
    dict_predictions,
    pids_to_plot,
    output_basis,
    pid_names,
    q_targets,
):
    """
    Generates a comparison grid for multiple NN models vs LHAPDF.
    """
    q2_values = [q**2 for q in q_targets]
    selected_q2 = []
    available_q2 = np.unique(input_q2grid)

    for q2_target in q2_values:
        closest_val = available_q2[np.abs(available_q2 - q2_target).argmin()]
        selected_q2.append(closest_val)

    n_pids = len(pids_to_plot)
    n_q2 = len(selected_q2)

    colors = [
        "#e41a1c",
        "#013661",
        "#4daf4a",
        "#984ea3",
        "#fbff00",
        "#00ffea",
        "#ff8d41",
    ]

    fig, axes = plt.subplots(
        nrows=n_pids * 2,
        ncols=n_q2,
        figsize=(18, 4 * n_pids),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1] * n_pids},
    )

    if n_pids == 1:
        axes = axes.reshape(2, n_q2)
    if n_q2 == 1:
        axes = axes.reshape(n_pids * 2, 1)

    for row_idx, pid in enumerate(pids_to_plot):
        idx_in_basis = output_basis.index(pid)
        main_row = row_idx * 2
        ratio_row = main_row + 1

        for col, q2_val in enumerate(selected_q2):
            q2_mask = input_q2grid == q2_val
            x_plot = input_xgrid[q2_mask]
            real_plot = output_data[q2_mask, idx_in_basis]

            sort_idx = np.argsort(x_plot)
            x_plot = x_plot[sort_idx]
            real_plot = real_plot[sort_idx]

            ax_main = axes[main_row, col]
            ax_ratio = axes[ratio_row, col]

            # --- Plot LHAPDF ---
            ax_main.plot(x_plot, real_plot, "k-", lw=2, label="LHAPDF", zorder=1)

            # --- Models ---
            for i, (model_name, preds) in enumerate(dict_predictions.items()):
                pred_plot = preds[q2_mask, idx_in_basis][sort_idx]
                color = colors[i % len(colors)]

                # Main Plot
                ax_main.plot(
                    x_plot, pred_plot, color=color, ls="--", lw=1.5, label=model_name
                )

                # Ratio Plot
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = pred_plot / real_plot
                ax_ratio.plot(x_plot, ratio, color=color, lw=1.2, alpha=0.8)

            # Plots config
            ax_main.set_xscale("log")
            ax_main.grid(True, which="both", ls=":", alpha=0.5)

            ax_ratio.axhline(1.0, color="black", linestyle="-", lw=1)
            ax_ratio.set_ylim(0.9, 1.1)
            ax_ratio.set_xscale("log")
            ax_ratio.grid(True, which="both", ls=":", alpha=0.5)

            if row_idx == 0:
                ax_main.set_title(f"$Q = {np.sqrt(q2_val):.2f}$ GeV", fontsize=14)
            if col == 0:
                ax_main.set_ylabel(
                    f"{pid_names[row_idx]}\n$xf(x, Q^2)$", fontweight="bold"
                )
                ax_ratio.set_ylabel("Models/Data", fontsize=9)
            if row_idx == n_pids - 1:
                ax_ratio.set_xlabel("$x$", fontsize=12)
            if row_idx == 0 and col == n_q2 - 1:
                ax_main.legend(loc="upper right", fontsize="small")

    plt.suptitle(
        "Comparative Analysis: Neural Network Architectures vs LHAPDF",
        fontsize=22,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.subplots_adjust(hspace=0.2)

    return fig


def plot_pdf_valence_multi(
    input_xgrid, input_q2grid, output_data, dict_predictions, q_values_to_plot
):
    """
    Generates a comparison of valence, gluon, and sea distributions for MULTIPLE models.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    q2_values_to_plot = [q**2 for q in q_values_to_plot]
    n_plots = len(q2_values_to_plot)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Flavor mapping based on your output_basis
    idx = {"u_bar": 2, "d_bar": 3, "g": 4, "d": 5, "u": 6, "s": 7, "c": 8}

    # We defined line styles to distinguish the models (M1 to M7)
    model_styles = [
        "--",  # M1
        "-.",  # M2
        ":",  # M3
        (0, (3, 5, 1, 5)),  # M4
        (0, (5, 5)),  # M5
        (0, (1, 1)),  # M6
        (0, (5, 1, 1, 1)),  # M7
    ]
    available_q2 = np.unique(input_q2grid)

    for i, q2_target in enumerate(q2_values_to_plot):
        if i >= len(axes):
            break
        ax = axes[i]

        closest_q2 = available_q2[np.abs(available_q2 - q2_target).argmin()]
        mask_q2 = input_q2grid == closest_q2

        x_s = input_xgrid[mask_q2]
        sort_idx = np.argsort(x_s)
        x_s = x_s[sort_idx]

        y_t_s = output_data[mask_q2][sort_idx]

        # --- 1. Draw LHAPDF Reference (Thick solid line behind) ---
        flavors_data = [
            ((y_t_s[:, idx["u"]] - y_t_s[:, idx["u_bar"]]), r"$u_v$", "#0004ff"),
            ((y_t_s[:, idx["d"]] - y_t_s[:, idx["d_bar"]]), r"$d_v$", "#ffa600"),
            (y_t_s[:, idx["g"]] / 10, r"$g/10$", "#ff0000"),
            (y_t_s[:, idx["u_bar"]], r"$\bar{u}$", "#c300ff"),
            (y_t_s[:, idx["d_bar"]], r"$\bar{d}$", "#0C6600"),
        ]

        for val, label, color in flavors_data:
            ax.plot(
                x_s,
                val,
                label=f"{label} (LHAPDF)",
                color=color,
                lw=2.5,
                alpha=0.3,
                zorder=1,
            )

        # --- 2. Draw each model from the dictionary ---
        for m_idx, (m_name, preds) in enumerate(dict_predictions.items()):
            y_p_s = preds[mask_q2][sort_idx]
            style = model_styles[m_idx % len(model_styles)]

            # We calculate combinations for this specific model
            uv_p = y_p_s[:, idx["u"]] - y_p_s[:, idx["u_bar"]]
            dv_p = y_p_s[:, idx["d"]] - y_p_s[:, idx["d_bar"]]
            g10_p = y_p_s[:, idx["g"]] / 10
            ubar_p = y_p_s[:, idx["u_bar"]]
            dbar_p = y_p_s[:, idx["d_bar"]]

            # We draw the lines of the model
            ax.plot(x_s, uv_p, color="#0004ff", ls=style, lw=1.2, zorder=2)
            ax.plot(x_s, dv_p, color="#ffa600", ls=style, lw=1.2, zorder=2)
            ax.plot(x_s, g10_p, color="#ff0000", ls=style, lw=1.2, zorder=2)
            ax.plot(x_s, ubar_p, color="#c300ff", ls=style, lw=1.2, zorder=2)
            ax.plot(x_s, dbar_p, color="#0C6600", ls=style, lw=1.2, zorder=2)

            if i == 0:
                ax.plot([], [], color="gray", ls=style, label=m_name)

        ax.set_xscale("log")
        ax.set_xlim([1e-3, 1])
        ax.set_ylim([0, 1.1])
        ax.set_title(rf"PDFs at $Q = {np.sqrt(closest_q2):.2f}$ GeV", fontsize=14)
        ax.grid(True, which="both", ls=":", alpha=0.5)

        if i == 0:
            ax.legend(loc="upper right", fontsize="7", ncol=2)

    plt.tight_layout()
    return fig


def generate_multi_model_physics_report(
    x, y_true, dict_predictions, f_map, q2_value, export_path=None
):
    """
    Calculates sum rules (Momentum, u, d, s, c valence) for multiple models
    and generates a comparative table.
    """
    if len(x) == 0:
        print(f"Error: No data available for Q2 = {q2_value}.")
        return None

    # Avoid division by zero for valence rules integral (integral of (q - qbar)/x dx)
    safe_x = np.where(x == 0, 1e-10, x)

    # 1. --- REFERENCE CALCULATIONS (LHAPDF / Ground Truth) ---
    # Momentum Rule: Integral of sum[x * f_i(x)] dx
    mom_t = np.trapz(np.sum([y_true[:, i] for i in f_map.values()], axis=0), x)

    # Helper for Valence Rules: Integral of [q(x) - q_bar(x)] / x * x dx = Integral of (q - q_bar) dx
    # Since your network outputs xf(x), we calculate: Integral of (xf - xf_bar) / x dx
    def get_val_true(flavor):
        return np.trapz(
            (y_true[:, f_map[flavor]] - y_true[:, f_map[f"{flavor}_bar"]]) / safe_x, x
        )

    ut_t = get_val_true("u")
    dt_t = get_val_true("d")
    st_t = get_val_true("s")
    ct_t = get_val_true("c")

    # 2. --- TABLE DATA STRUCTURE ---
    # Defining rows for the physical constraints
    report_data = [
        {"Sum Rule": "Total Momentum", "Target": 1.0, "LHAPDF": mom_t},
        {"Sum Rule": "u Valence", "Target": 2.0, "LHAPDF": ut_t},
        {"Sum Rule": "d Valence", "Target": 1.0, "LHAPDF": dt_t},
        {"Sum Rule": "s Valence", "Target": 0.0, "LHAPDF": st_t},
        {"Sum Rule": "c Valence", "Target": 0.0, "LHAPDF": ct_t},
    ]

    # 3. --- PER-MODEL CALCULATIONS ---
    for m_name, y_pred in dict_predictions.items():
        # Momentum sum calculation for the current model
        mom_p = np.trapz(np.sum([y_pred[:, i] for i in f_map.values()], axis=0), x)

        # Valence sum calculations for the current model
        def get_val_pred(flavor):
            return np.trapz(
                (y_pred[:, f_map[flavor]] - y_pred[:, f_map[f"{flavor}_bar"]]) / safe_x,
                x,
            )

        up_p = get_val_pred("u")
        dp_p = get_val_pred("d")
        sp_p = get_val_pred("s")
        cp_p = get_val_pred("c")

        # Map results to the corresponding rows in the report
        report_data[0][m_name] = mom_p
        report_data[1][m_name] = up_p
        report_data[2][m_name] = dp_p
        report_data[3][m_name] = sp_p
        report_data[4][m_name] = cp_p

    df = pd.DataFrame(report_data)

    # 4. --- LATEX EXPORT (Ready for Overleaf/Thesis) ---
    if export_path and export_path.endswith(".tex"):
        df.to_latex(
            export_path,
            index=False,
            float_format="%.4f",
            caption=f"Comparison of Physical Sum Rules at $Q={np.sqrt(q2_value):.2f}$ GeV",
            label="tab:sum_rules_multi_model",
            escape=False,
        )
        print(f"LaTeX table successfully exported to: {export_path}")

    # 5. --- VISUALIZATION AND STYLING ---
    print(
        f"\n--- Multi-Model Physical Consistency Report (Q = {np.sqrt(q2_value):.2f} GeV) ---"
    )

    # Return styled DataFrame for Jupyter display
    return df.style.format(precision=4)
