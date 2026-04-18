import tensorflow as tf


## Layers ##
class InputScaling(tf.keras.layers.Layer):
    """ "Logarithmic scaling of the input data."""

    def call(self, inputs):
        # inputs[:,0] it's x and inputs[:,1] it's Q2
        x = inputs[:, 0:1]
        q2 = inputs[:, 1:2]

        log_x = tf.math.log(x + 1e-10)  # Add a small constant to avoid log(0)
        log_q2 = tf.math.log(q2 + 1e-10)  # Add a small constant to avoid log(0)

        return tf.concat([x, log_x, log_q2], axis=-1)


class Preprocessing(tf.keras.layers.Layer):
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

    def call(self, inputs):
        x = inputs[:, 0:1]  # Extract x from the inputs
        # We need to apply (1-x)^(1+beta_i) to each output i, so we will expand the dimensions of x and beta to do this multiplication correctly.
        return (1.0 - x) ** (self._beta + 1.0)


# This function is used to dynamically build hidden layers based on the provided architecture in Model2.
def build_hidden_layer(input_tensor, layer_dims, activation="tanh"):
    """
    Dynamically chains layers.
    input_tensor: The tensor from the previous layer.
    layer_dims: List with the number of neurons, e.g., [64, 64, 64].
    """
    x = input_tensor
    for i, units in enumerate(layer_dims):
        x = tf.keras.layers.Dense(units, activation=activation, name=f"dense_{i + 1}")(
            x
        )
    return x
