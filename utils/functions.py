import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from IPython.display import display

## --- Pid to latex ---
# Function to translate the pid of the partons to their corresponding LaTeX symbols.


def pid_to_latex(pid):
    translate = {21: "g", 1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t"}
    flav = translate[abs(pid)]
    if pid < 0:
        flav = rf"\bar{{{flav}}}"
    return f"${flav}$"


## --- plot pdf comparison --- ##
# Function to plot differents PDFs


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


def plot_pdfs(
    x_inputs,
    q2_inputs,
    output_data,
    predictions,
    output_basis,
    pids_to_plot=None,
    q2_to_plot=None,
    figsize=(18, 15),
    title="Model Comparative: NN vs LHAPDF",
):
    """
    Generates a grid of subplots comparing neural network predictions
    against ground truth values (LHAPDF) for specific Q^2 values.
    """

    # 1. PIDs and labels configuration
    if pids_to_plot is None:
        pids_to_plot = [21, 1, 2, -1, 3, -2, -3, 4, -4]

    pid_names = [pid_to_latex(p) for p in pids_to_plot]

    # 2. Select 3 representative Q^2 values for plotting (e.g., low, medium, high)
    unique_q2_vals = np.unique(q2_inputs)
    indices_q2 = np.linspace(0, len(unique_q2_vals) - 1, 3, dtype=int)
    selected_q2 = unique_q2_vals[indices_q2]

    # 3. Figure configuration
    n_rows = len(pids_to_plot)
    n_cols = len(selected_q2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)

    # Ensure axes is always a 2D array
    if n_rows == 1 or n_cols == 1:
        axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for row, pid in enumerate(pids_to_plot):
        idx_in_basis = output_basis.index(pid)

        for col, q2_val in enumerate(selected_q2):
            ax = axes[row, col]

            # Mask to select specific data points
            # Note: We use np.isclose for float comparison to avoid precision issues
            q2_mask = np.isclose(q2_inputs, q2_val)

            x_plot = x_inputs[q2_mask]
            real_plot = output_data[q2_mask, idx_in_basis]
            pred_plot = predictions[q2_mask, idx_in_basis]

            if len(x_plot) == 0:
                ax.text(0.5, 0.5, f"Q²={q2_val} not found", ha="center")
                continue

            # Sort by x to prevent "zig-zags"
            sort_idx = np.argsort(x_plot)
            x_plot = x_plot[sort_idx]
            real_plot = real_plot[sort_idx]
            pred_plot = pred_plot[sort_idx]

            # Plotting
            ax.plot(x_plot, real_plot, "k-", lw=2, label="LHAPDF (Target)")
            ax.plot(x_plot, pred_plot, "r--", lw=2, label="NN Model")

            ax.set_xscale("log")

            if row == 0:
                ax.set_title(f"$Q^2 = {q2_val:.2f}$ GeV$^2$", fontsize=14, pad=15)

            if col == 0:
                ax.set_ylabel(
                    f"{pid_names[row]}\n$x f(x, Q^2)$", fontsize=12, fontweight="bold"
                )

            if row == n_rows - 1:
                ax.set_xlabel("$x$", fontsize=12)

            ax.grid(True, which="both", ls=":", alpha=0.5)

            if row == 0 and col == n_cols - 1:
                ax.legend()

    plt.suptitle(title, fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def get_final_report(x, y_true, y_pred, f_map):
    # Avoid division by zero at x=0
    safe_x = np.where(x == 0, 1e-10, x)

    # --- TOTAL MOMENTUM SUM RULE ---
    # Integral of sum[x*f(x)] dx
    mom_t = np.trapz(np.sum([y_true[:, i] for i in f_map.values()], axis=0), x)
    mom_p = np.trapz(np.sum([y_pred[:, i] for i in f_map.values()], axis=0), x)

    # --- VALENCE SUM RULES (Integrals of [q(x) - q_bar(x)] dx) ---
    # Valence u: (x*u - x*u_bar) / x
    uv_t = np.trapz((y_true[:, f_map["u"]] - y_true[:, f_map["u_bar"]]) / safe_x, x)
    uv_p = np.trapz((y_pred[:, f_map["u"]] - y_pred[:, f_map["u_bar"]]) / safe_x, x)

    # Valence d: (x*d - x*d_bar) / x
    dv_t = np.trapz((y_true[:, f_map["d"]] - y_true[:, f_map["d_bar"]]) / safe_x, x)
    dv_p = np.trapz((y_pred[:, f_map["d"]] - y_pred[:, f_map["d_bar"]]) / safe_x, x)

    # Valence s: (x*s - x*s_bar) / x (Expected: 0)
    sv_t = np.trapz((y_true[:, f_map["s"]] - y_true[:, f_map["s_bar"]]) / safe_x, x)
    sv_p = np.trapz((y_pred[:, f_map["s"]] - y_pred[:, f_map["s_bar"]]) / safe_x, x)

    # Valence c: (x*c - x*c_bar) / x (Expected: 0)
    cv_t = np.trapz((y_true[:, f_map["c"]] - y_true[:, f_map["c_bar"]]) / safe_x, x)
    cv_p = np.trapz((y_pred[:, f_map["c"]] - y_pred[:, f_map["c_bar"]]) / safe_x, x)

    # --- DATAFRAME CONSTRUCTION ---
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
            "LHAPDF (Target)": [mom_t, uv_t, dv_t, sv_t, cv_t],
            "NN (Prediction)": [mom_p, uv_p, dv_p, sv_p, cv_p],
        }
    )

    # Error Metrics
    df["Absolute Error"] = np.abs(df["LHAPDF (Target)"] - df["NN (Prediction)"])
    # Fidelity calculation with safety clip for null valences
    df["Fidelity (%)"] = (
        1 - (df["Absolute Error"] / np.clip(np.abs(df["LHAPDF (Target)"]), 1e-4, None))
    ) * 100

    return df


def display_physics_report(x_f, y_true_f, y_pred_f, idx, q2_value):
    """
    Generates and displays the physical consistency report with visual formatting.
    """

    if len(x_f) > 0:
        # 1. Generate the base DataFrame
        report_df = get_final_report(x_f, y_true_f, y_pred_f, idx)

        # 2. Print header with Q (square root of Q2)
        print(f"Physical Consistency Report (Q = {np.sqrt(q2_value):.2f} GeV)")

        # 3. Apply and display the styling
        styled_report = report_df.style.format(
            {
                "Theoretical": "{:.1f}",
                "LHAPDF (Target)": "{:.4f}",
                "NN (Prediction)": "{:.4f}",
                "Absolute Error": "{:.2e}",
                "Fidelity (%)": "{:.2f}%",
            }
        ).background_gradient(subset=["Fidelity (%)"], cmap="RdYlGn", vmin=0, vmax=100)

        display(styled_report)
    else:
        print(f"Error: Empty mask for Q2 = {q2_value}. Please check the dataset.")


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
