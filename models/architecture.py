import tensorflow as tf
from .layers import InputScaling, Preprocessing


## Model used for the first objective: predict NN(x,Q2) = LHAPDF(x,Q2) ##
def Model1(input_shape=(2,), noutput=None):

    # Input: (x, Q2)
    # Output: NN(x, Q2)
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Logarithmic scaling of the input data.
    scaled = InputScaling()(inputs)
    x = tf.keras.layers.Dense(64, activation="tanh")(scaled)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)

    # Output layer with linear activation to predict the PDF values for each PID
    pdf_raw = tf.keras.layers.Dense(noutput, activation="linear")(x)

    # Ensure the limit of x = 0 as x approaches 1.

    preproc = Preprocessing(noutput)(inputs)
    final_results = tf.keras.layers.Multiply()([pdf_raw, preproc])
    return tf.keras.Model(inputs=inputs, outputs=final_results)



def Model2(input_shape=(2,), noutput=None):

    # Input: (x, Q2)
    # Output: NN(x, Q2)
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Logarithmic scaling of the input data.
    scaled = InputScaling()(inputs)
    x = tf.keras.layers.Dense(256, activation="tanh")(scaled)
    x = tf.keras.layers.Dense(256, activation="tanh")(x)
    x = tf.keras.layers.Dense(256, activation="tanh")(x)

    # Output layer with linear activation to predict the PDF values for each PID
    pdf_raw = tf.keras.layers.Dense(noutput, activation="linear")(x)

    # Ensure the limit of x = 0 as x approaches 1.

    preproc = Preprocessing(noutput)(inputs)
    final_results = tf.keras.layers.Multiply()([pdf_raw, preproc])
    return tf.keras.Model(inputs=inputs, outputs=final_results)

def Model3(input_shape=(2,), noutput=None):

    # Input: (x, Q2)
    # Output: NN(x, Q2)
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Logarithmic scaling of the input data.
    scaled = InputScaling()(inputs)
    x = tf.keras.layers.Dense(64, activation="tanh")(scaled)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)

    # Output layer with linear activation to predict the PDF values for each PID
    pdf_raw = tf.keras.layers.Dense(noutput, activation="linear")(x)

    # Ensure the limit of x = 0 as x approaches 1.

    preproc = Preprocessing(noutput)(inputs)
    final_results = tf.keras.layers.Multiply()([pdf_raw, preproc])
    return tf.keras.Model(inputs=inputs, outputs=final_results)