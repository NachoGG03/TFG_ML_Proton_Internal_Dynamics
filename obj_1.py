### Final Project - PDF Modeling with Neural Networks ###
# This code is part of a project to model Parton Distribution Functions (PDFs) using neural networks. The goal is to train a model to predict the PDFs based on input variables x and Q^2, which represent the momentum fraction and energy scale, respectively.
# In this script we will see the results of the main model and make plots for upcoming discussions.

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from utils.functions import (
    pid_to_latex,
    plot_LHAPDF,
    plot_pdf_comparison,
    plot_pdf_valence,
    generate_physics_report
)


### Load the data ###
# The data is stored in a .npy file, which contains the following arrays:
# - pdf_target: the target probability density function (PDF) values for each sample
# - x_q2_inputs: the input values for the variable x and Q^2 for each sample
# - pids_info: the particle IDs (PIDs) for each sample
data_path = "data/"

try:
    x_q2_inputs = np.load(os.path.join(data_path, "x_q2_inputs.npy"))
    val_pdf = np.load(os.path.join(data_path, "pdf_targets.npy"))
    pids = list(np.load(os.path.join(data_path, "pids_info.npy")))

    print("Data successfully loaded. Data of NNPDF40_nnlo_as_01180 set is being used.")
    print(f"Inputs Format (X): {x_q2_inputs.shape}  -> [points, (x, Q2)]")
    print(f"Targets Format (y): {val_pdf.shape} -> [values, flavors]")
    print(f"PIDs included: {pids}")

except FileNotFoundError:
    print(
        "Error: Data files not found. Please ensure the .npy files are in the 'data/' directory."
    )
    exit(1)

input_xgrid = x_q2_inputs[:, 0]
input_q2grid = x_q2_inputs[:, 1]

# The model was trained in another pc and the predictions was compile in a script .npz.
# Loading the predictions:
outputs_path = "outputs/"
try:
    predictions_load = np.load(os.path.join(outputs_path, "main_model.npz"))
    predictions = predictions_load["predictions"]
    print(f"Predictions: {predictions.shape} -> [values, flavors]")
except FileNotFoundError:
    print(
        "Error: Data file not found. Please ensure the .npz file are in the 'outputs/' directory."
    )
    exit(1)

# Make the grid with the PDF values to plot:
pid_cols = {pid: i for i, pid in enumerate(pids)}
output_basis = [
    -4,
    -3,
    -2,
    -1,
    21,
    1,
    2,
    3,
    4,
]
noutput = len(output_basis)  # Number of flavors (columns in val_pdf)
output_data = np.zeros((len(val_pdf), noutput))

pdf_grid = defaultdict(list)

for j, pid in enumerate(output_basis):
    col_idx = pid_cols[pid]
    output_data[:, j] = val_pdf[:, col_idx]

output_data = np.array(output_data)

# --- Save path --- #
results_folder = "results/"

## ----- PLOTS ----- ###
# At first we will inspect how the PDFs looks like
en = [91]  # Energie value to plot
flavours_to_pid = [21, 2, 1, -1]  # Wich PDFs we want to see
plot_LHAPDF(
    input_xgrid=input_xgrid,
    input_q2grid=input_q2grid,
    val_pdf=val_pdf,
    pids=pids,
    scales=en,
    flavour_select=flavours_to_pid,
)
plt.show()
plt.close()

# Then we will see how the model fix the data:
q_values = [5, 91, 200, 1000]

plot_pdf_comparison(
    input_xgrid = input_xgrid,
    input_q2grid = input_q2grid,
    output_data=output_data,
    predictions=predictions,
    pids_to_plot=output_basis,
    output_basis=output_basis,
    pid_names=[pid_to_latex(p) for p in output_basis],
    q_targets=q_values,
)
plt.show()
plt.close()

plot_pdf_valence(
    input_xgrid=input_xgrid,
    input_q2grid=input_q2grid,
    output_data=output_data,
    predictions=predictions,
    q_values_to_plot=q_values
)
path=os.path.join(results_folder, "fig_4.png")
plt.savefig(path,dpi=300,bbox_inches='tight')
plt.show()
plt.close()

# Next, we will verify whether the neural network has been able to learn certain physical constraints on its own.
# 1. Input Configuration
target_q = 91
target_q2 = target_q**2

# Index mapping for flavors and anti-flavors
idx = {
    "u": 6, "u_bar": 2, 
    "d": 5, "d_bar": 3, 
    "s": 7, "s_bar": 1, 
    "c": 8, "c_bar": 0, 
    "g": 4
}

# 2. Robust Q2 Scale Selection
available_q2 = np.unique(input_q2grid)
closest_q2 = available_q2[np.abs(available_q2 - target_q2).argmin()]
mask = (input_q2grid.flatten() == closest_q2)

# 3. Data Extraction and Sorting by x
x_vals = input_xgrid.flatten()[mask]
sort_indices = np.argsort(x_vals)
x_f = x_vals[sort_indices]
y_true_f = output_data[mask][sort_indices]
y_pred_f = predictions[mask][sort_indices]

report = generate_physics_report(
    x = x_f,
    y_true = y_true_f,
    y_pred = y_pred_f,
    f_map = idx,
    q2_value = closest_q2,
    export_path = "results/table_momentum_rules.tex"
)

