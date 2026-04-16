## --- Pid to latex ---
# Function to translate the pid of the partons to their corresponding LaTeX symbols.


def pid_to_latex(pid):
    translate = {21: "g", 1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t"}
    flav = translate[abs(pid)]
    if pid < 0:
        flav = rf"\bar{{{flav}}}"
    return f"${flav}$"

import matplotlib.pyplot as plt
import numpy as np

## --- plot pdf comparison --- ##
# Function to plot differents PDFs 

def plot_pdf_comparison(x_inputs, q2_inputs, output_data, predictions, output_basis, 
                        pids_to_plot=None, q2_to_plot=None, figsize=(18, 15), 
                        title="Model Comparative: NN vs LHAPDF"):
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
                ax.text(0.5, 0.5, f"Q²={q2_val} not found", ha='center')
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
                ax.set_ylabel(f"{pid_names[row]}\n$x f(x, Q^2)$", fontsize=12, fontweight="bold")
            
            if row == n_rows - 1:
                ax.set_xlabel("$x$", fontsize=12)

            ax.grid(True, which="both", ls=":", alpha=0.5)
            
            if row == 0 and col == n_cols - 1:
                ax.legend()

    plt.suptitle(title, fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig