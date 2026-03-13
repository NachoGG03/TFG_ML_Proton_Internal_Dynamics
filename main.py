### FITTING A PDF AT A FIXED SCALE (91 GeV Z BOSSON MASS) ###

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from collections import defaultdict
from lhapdf import mkPDF, setVerbosity
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from utils.functions import pid_to_latex

tf.keras.backend.clear_session()
setVerbosity(0)

## Reading the DATA from LHAPDF, we will use the NNPDF40_nnlo_as_01180 set.
pdf_set = "NNPDF40_nnlo_as_01180/0"
pdf_target = mkPDF(pdf_set)

## Defining the scale at which we want to fit the PDF, in this case the Z boson mass.
q0 = 91.1876  # GeV
npoints = int(1e5)

# We will use a logarithmic grid in x to capture the behavior at small x, and a linear grid at large x to capture the behavior near 1.
xgrid = np.concatenate(
    (np.logspace(-5, -1, npoints // 2), np.linspace(0.1, 1, npoints // 2))
)
input_xgrid = xgrid.reshape(-1, 1)

# This will give us a list of dictionaries, where each dictionary contains the PDF values for all partons at a given x and Q^2.
pdf_grid_all = pdf_target.xfxQ2(xgrid, np.ones_like(xgrid) * q0**2)

# We will plot some PDFs to inspect them visually.
flavour_select = [1, 2, 21, -1]  # Gluon, down, up, anti-up
pdf_grid = defaultdict(list)

for point in pdf_grid_all:
    for pid in flavour_select:
        pdf_grid[pid_to_latex(pid)].append(point[pid])

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f"PDF plots for {pdf_set} at Q={q0} GeV", fontsize=16)
for i, (parton, val) in enumerate(pdf_grid.items()):
    plt.subplot(2, 2, i + 1)

    plt.plot(xgrid, val, label=f"{parton} PDF")
    plt.legend
    plt.xscale("log")
    if i > 1:
        plt.xlabel("x")
    plt.ylabel(f"$x{parton}(x)$")

## Now we will focus on fitting the gluon PDF.

target_pid = 21  # Gluon
name = pid_to_latex(target_pid)
parton_data = np.array([pdf_grid[target_pid] for pdf_grid in pdf_grid_all]).reshape(
    -1, 1
)

# Normalization trick: We will normalize the PDF values to be between 0 and 1, to help the neural network learn better.
max_val = np.max(parton_data)
parton_data_norm = parton_data / max_val

# Now parton_data_norm is between 0 and 1, which should make the training of the neural network more stable.

# We can observe that the input data involves different scales and wich require some form of preprocessing of the input.

c, b = np.histogram(xgrid)
cl, bl = np.histogram(np.log(xgrid))

plt.subplots(1, 2, figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.title("Linear")
plt.stairs(c, b, fill=True)
plt.xlabel("x")
plt.ylabel("N")
plt.subplot(1, 2, 2)
plt.title("Log")
plt.stairs(cl, bl, fill=True)
plt.xlabel("log(x)")
plt.ylabel("N")

# ---------------------------------------------------------------------------------------------------------------


## CUSTOM LAYERS
# We know that the behavior of the partons must go to 0 as x goes to 1, so we will include this information in the model by multiplying the output of the neural network by (1-x)^beta, where beta is a trainable parameter that will allow the model to learn the correct behavior at large x.
# Additionally, we will include the logarithm of x as an input to the model, since the PDFs have a strong dependence on log(x) at small x.
class InputScaling(tf.keras.layers.Layer):
    def call(self, x):
        return tf.concat([x, tf.math.log(x + 1e-10)], axis=-1)


class Preprocessing(tf.keras.layers.Layer):
    def build(self, input_shape):
        self._beta = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            name="beta",
            constraint=tf.keras.constraints.non_neg(),
        )

    def call(self, x):
        return (1.0 - x) ** (self._beta + 1.0)


## MODEL ARCHITECTURE


def generate_model(input_shape=(1,)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Hemos aumentado un poco las unidades (de 14 a 32) para capturar mejor la curvatura
    scaled = InputScaling()(inputs)
    x = tf.keras.layers.Dense(32, activation="tanh")(scaled)
    x = tf.keras.layers.Dense(32, activation="tanh")(x)
    x = tf.keras.layers.Dense(32, activation="tanh")(x)

    pdf_raw = tf.keras.layers.Dense(1, activation="linear")(x)
    preproc = Preprocessing()(inputs)

    final_results = tf.keras.layers.Multiply()([pdf_raw, preproc])

    return tf.keras.models.Model(inputs=inputs, outputs=final_results)


model = generate_model()
# Now we shut down the learning rate for a more fine-tuned fit at the end.
model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.0005), loss="mse")

history = model.fit(
    input_xgrid,
    parton_data_norm,
    epochs=15,
    validation_split=0.3,
    batch_size=512,
    verbose=1,
)

## RESULTS AND PLOTS ##
predictions_norm = model.predict(input_xgrid)
# We need to multiply the predictions by the maximum value to compare with the original data, since we trained with normalized data.
predictions = predictions_norm * max_val

plt.figure(figsize=(10, 5))

# Plot Lineal
plt.subplot(1, 2, 1)
plt.plot(xgrid, parton_data, label="g")
plt.plot(xgrid, predictions, label="NN", linestyle="--")
plt.title("Escala Lineal")
plt.xlabel("x")
plt.ylabel(rf"$x{name}(x, Q_{0})$")
plt.legend()

# Plot Log
plt.subplot(1, 2, 2)
plt.plot(xgrid, parton_data, label="g")
plt.plot(xgrid, predictions, label="NN", linestyle="--")
plt.xscale("log")
plt.title("Escala Log")
plt.xlabel("x")
plt.ylabel(rf"$x{name}(x, Q_{0})$")
plt.legend()

### MULTIFLAVOR FIT ###

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
]  # anti-charm, anti-strange, anti-up, anti-down, gluon, down, up, strange, charm
noutput = len(output_basis)

output_data = np.zeros((len(pdf_grid_all), noutput))

for i, pdf_grid in enumerate(pdf_grid_all):
    for j, pid in enumerate(output_basis):
        output_data[i, j] = pdf_grid[pid]

output_data = np.array(output_data)

# Normalize each column by its maximum value
max_vals = np.max(output_data, axis=0)
output_data_norm = output_data / max_vals


# Now the beta parameter in the Preprocessing layer should be a vector of length noutput, and we will multiply each output by its corresponding (1-x)^beta_i.
class PreprocessingMulti(tf.keras.layers.Layer):
    def __init__(self, noutput, **kwargs):
        super().__init__(**kwargs)
        self.noutput = noutput

    def build(self, input_shape):
        # Beta is now a vector of length noutput, one for each output flavor
        self._beta = self.add_weight(
            shape=(self.noutput,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            name="beta",
            constraint=tf.keras.constraints.non_neg(),
        )

    def call(self, x):
        # We need to apply (1-x)^(1+beta_i) to each output i, so we will expand the dimensions of x and beta to do this multiplication correctly.
        return (1.0 - x) ** (self._beta + 1.0)


# The model architecture will be similar to the single flavor case, but now we will have noutput outputs, and we will use the PreprocessingMulti layer to apply the correct behavior at large x for each flavor.
def pdf_model_multiflavour(input_shape=(1,)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    scaled = InputScaling()(inputs)
    x = tf.keras.layers.Dense(64, activation="tanh")(scaled)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)

    raw_pdf_multiflavour = tf.keras.layers.Dense(noutput, activation="linear")(x)
    preproc_multiflavour = PreprocessingMulti(noutput)(inputs)

    final_results_multiflavour = tf.keras.layers.Multiply()(
        [raw_pdf_multiflavour, preproc_multiflavour]
    )

    return tf.keras.models.Model(inputs=inputs, outputs=final_results_multiflavour)


model_multiflavour = pdf_model_multiflavour()
model_multiflavour.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=0.0005), loss="mse"
)
history_multiflavour = model_multiflavour.fit(
    input_xgrid, output_data_norm, epochs=15, validation_split=0.3, batch_size=512
)

predictions_multiflavour_norm = model_multiflavour.predict(input_xgrid)
predictions_multiflavour = predictions_multiflavour_norm * max_vals

plt.subplots(2, 2, figsize=(12, 12))
for i, pid in enumerate([1, 2, 21, -1]):
    idx = output_basis.index(pid)
    name = pid_to_latex(pid)

    plt.subplot(2, 2, i + 1)
    plt.plot(xgrid, output_data[:, idx], label=name)
    plt.plot(xgrid, predictions_multiflavour[:, idx], label="NN", linestyle="--")
    plt.xscale("log")
    plt.xlabel("x")
    plt.ylabel(rf"$x{name}(x, Q_{0})$")
    plt.legend()

plt.show()
