# In this script, we will train our model with fewer data points (1e8–1e6).
# Furthermore, we will train models with different parameters to observe how their behavior varies.
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from tensorflow import keras
from matplotlib import pyplot as plt
from utils.functions import pid_to_latex, plot_pdf_comparison_multi, plot_pdf_valence_multi, generate_multi_model_physics_report
from models.layers import InputScaling, Preprocessing
from models.architecture import Model1, Model2, Model3

from IPython.display import display
tf.keras.backend.clear_session()

# Load the data for this part:
data_path = "data/"

try:
    x_q2_inputs = np.load(os.path.join(data_path, "x_q2_inputs_training.npy"))
    val_pdf = np.load(os.path.join(data_path, "pdf_targets_training.npy"))
    pids = np.load(os.path.join(data_path, "pids_info_training.npy"))
    print("Data successfully loaded. Data of NNPDF40_nnlo_as_01180 set is being used.")
    print(f"Inputs Format (X): {x_q2_inputs.shape}  -> [points, (x, Q2)]")
    print(f"Targets Format (y): {val_pdf.shape} -> [values, flavors]")
    print(f"PIDs included: {pids}")
except FileNotFoundError:
    print(
        "Error: Data files not found. Please ensure the .npy files are in the 'data/' directory."
    )
    exit(1)

input_xgrid = x_q2_inputs[:,0]
input_q2grid = x_q2_inputs[:,1]

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
print(f"Number of samples: {len(val_pdf)}, Number of flavors: {noutput}")

pdf_grid = defaultdict(list)

for j, pid in enumerate(output_basis):
    col_idx = pid_cols[pid]
    output_data[:, j] = val_pdf[:, col_idx]

output_data = np.array(output_data)

# Normalize each column by its maximum value
max_vals = np.max(output_data, axis=0)
output_data_norm = output_data / max_vals
## --- Training the models --- ##
# We will use the same seed for train all the models.

def reset_seeds(seed=42):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seeds reset to {seed} and session cleared.")

# --- SATART OF TRAINING LOOP --- #
reset_seeds(42)
model1 = Model1(input_shape=(2,), noutput=noutput)
model1.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.0005), loss="mse")
history = model1.fit(
    x_q2_inputs, output_data_norm, epochs=500, batch_size=1048,validation_split=0.3
)
predictions_norm1 = model1.predict(x_q2_inputs)
predictions1 = predictions_norm1 * max_vals
model1.summary()

reset_seeds(42)
model2 = Model1(input_shape=(2,), noutput=noutput)
model2.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.0005), loss="mse")
history = model2.fit(
    x_q2_inputs, output_data_norm, epochs=1000, batch_size=1048,validation_split=0.3
)
predictions_norm2 = model2.predict(x_q2_inputs)
predictions2 = predictions_norm2 * max_vals
model2.summary()

reset_seeds(42)
model3 = Model2(input_shape=(2,), noutput=noutput)
model3.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.0005), loss = "mse")
history = model3.fit(
    x_q2_inputs, output_data_norm, epochs=500, batch_size=1048,validation_split=0.3
)
predictions_norm3 = model3.predict(x_q2_inputs)
predictions3 = predictions_norm3 * max_vals
model3.summary()

reset_seeds(42)
model4 = Model3(input_shape=(2,), noutput=noutput)
model4.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.0005), loss = "mse")
history = model3.fit(
    x_q2_inputs, output_data_norm, epochs=500, batch_size=1048,validation_split=0.3
)
predictions_norm4 = model4.predict(x_q2_inputs)
predictions4 = predictions_norm4 * max_vals
model4.summary()

models = {
    "Model 1 (Base)": predictions1,
    "Model 2 (Epochs)": predictions2,
    "Model 3 (Deeper)": predictions3,
    "Model 4 (Wider)": predictions4 
}

# --- END OF TRAINING LOOP --- #

## --- RESULTS --- ##

q_values = [5,91,200,1000]
names = [pid_to_latex(p) for p in output_basis]
results_folder = "results/multi model"

plot_pdf_comparison_multi(
    input_xgrid,
    input_q2grid,
    output_data,
    models,
    output_basis,
    output_basis,
    names,
    q_values
)
path1 = os.path.join(results_folder,"Fig1.png")
plt.savefig(path1, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

plot_pdf_valence_multi(
    input_xgrid,
    input_q2grid,
    output_data,
    models,
    q_values
)
path2 = os.path.join(results_folder,"Fig2.png")
plt.savefig(path2, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Next, we will verify if our NNs have been able to learn certain physical constraints on their own.
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

dict_predictions = {}

for model_name, preds in models.items():
    dict_predictions[model_name] = preds[mask][sort_indices]

report = generate_multi_model_physics_report(
    x_f,
    y_true_f,
    dict_predictions,
    idx,
    closest_q2,
    export_path = "results/multi model/table_momentum_rules.tex"
)