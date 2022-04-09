from tensorflow_addons.losses import *
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers as l
K = tf.keras.backend


def binary_weighted_crossentropy(weight, use_tf=True):
    """ Compute weighted binary crossentropy using tensorflow routines from logits
    params:
        weight: int
            weight > 1 decreases false negative count and increases recall
            weight < 1 decreases false positive count and increases precision
    """
    def binary_weighted_crossentropy_internal(labels, logits):
        if 'int' in labels.dtype.name:
            labels = tf.cast(labels, 'float32')
        if use_tf:
            return tf.nn.weighted_cross_entropy_with_logits(labels, logits, weight)
        x, z, q = logits, labels, weight
        return (1 - z) * x + (1 + (q - 1) * z) * tf.math.log(1 + tf.exp(-x))
    return binary_weighted_crossentropy_internal

def weighted_binary_crossentropy(w1, w2):
    """
    Computes weighted binary crossentropy
    params:
        w1, w2: the weights for the two classes, (0, 1) respectively
    Usage:
     model.compile(
         loss=weighted_binary_crossentropy(0.7, 0.3), optimizer="adam", metrics=["accuracy"])
    """
    if tf.round(w1 + w2) != 1.0:
        raise ValueError("`recall_weight` and `spec_weight` must sum to 1.")
    @tf.autograph.experimental.do_not_convert
    def weighted_binary_crossentropy_internal(y_true, y_pred):
        # avoid absolute 0
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        ones   = tf.ones_like(y_true)
        mask   = tf.equal(y_true, ones)
        res, _ = tf.map_fn(lambda x: (tf.multiply(-tf.math.log(x[0]), w1) if x[1] is True
                                      else tf.multiply(-tf.math.log(1 - x[0]), w2), x[1]),
                           (y_pred, mask), dtype=(tf.float32, tf.bool))
        return res
    return weighted_binary_crossentropy_internal


def no_NaN_loss(fn):
    """
    Generic wrapper for any loss function.

    Adds small epsilon to predicted values for numerical stability
    """
    def no_NaN(x, y):
        return fn(x, y+1e-8)
    return no_NaN


# Bayesian classifier
# https://github.com/kyle-dorman/bayesian-neural-network-blogpost


# Bayesian categorical cross entropy.
# N data points, C classes, T monte carlo simulations
# true - true values. Shape: (N, C)
# pred_var - predicted logit values and variance. Shape: (N, C + 1)
# returns - loss (N,)
def bayesian_categorical_crossentropy(T, num_classes):
    iterable = K.variable(K.ones(T))
    def bayesian_categorical_crossentropy_internal(true, pred_var):
        true = tf.cast(true, 'float32')
        # shape: (N,)
        std = K.sqrt(pred_var[:, num_classes])
        # shape: (N,)
        variance = pred_var[:, num_classes]
        variance_depressor = K.exp(variance) - K.ones_like(variance)
        # shape: (N, C)
        pred = pred_var[:, 0:num_classes]
        # shape: (N,)
        undistorted_loss = K.categorical_crossentropy(true, pred, from_logits=True)
        # shape: (T,)
        dist = tfp.distributions.Normal(loc=K.zeros_like(std), scale=std)
        monte_carlo_results = K.map_fn(
            gaussian_categorical_crossentropy(
                true, pred, dist, undistorted_loss, num_classes),
            iterable,
            name='monte_carlo_results')
        variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss
        return variance_loss + undistorted_loss + variance_depressor
    return bayesian_categorical_crossentropy_internal
    

# for a single monte carlo simulation,
#   calculate categorical_crossentropy of
#   predicted logit values plus gaussian
#   noise vs true values.
# true - true values. Shape: (N, C)
# pred - predicted logit values. Shape: (N, C)
# dist - normal distribution to sample from. Shape: (N, C)
# undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
# num_classes - the number of classes. C
# returns - total differences for all classes (N,)
def gaussian_categorical_crossentropy(true,
                                      pred,
                                      dist,
                                      undistorted_loss,
                                      num_classes):
    def map_fn(i):
        std_samples = K.transpose(dist.sample(num_classes))
        distorted_loss = K.categorical_crossentropy(
            true, pred + std_samples, from_logits=True)
        diff = undistorted_loss - distorted_loss
        return -K.elu(diff)
    return map_fn


def add_aleatoric_loss(inputs, # input tensor
                       output, # output of current last layer in model (e.g. Dense(32))
                       num_classes,
                       logit_weight=0.2,
                       sigmoid_weight=1.0,
                       loss_weight=10,
                       init_lr=1e-4,
                       optim='adam',
                       n_monte_carlo=100):
    logits = l.Dense(num_classes)(output)
    variance_pre = l.Dense(1)(output)
    variance = l.Activation('softplus', name='variance')(variance_pre)
    logits_variance = l.concatenate([logits, variance], name='logits_variance')
    sigmoid_output = l.Activation('sigmoid', name='sigmoid_output')(logits)
    optimizer = tf.keras.optimizers.get(optim)
    optimizer = optimizer.from_config({'lr': init_lr})
    model = tf.keras.Model(inputs=inputs, outputs=[logits_variance, sigmoid_output])
    model.compile(
        optimizer=optimizer,
        loss={
        'logits_variance': bayesian_categorical_crossentropy(n_monte_carlo, 1),
        'sigmoid_output': binary_weighted_crossentropy(loss_weight)},
        metrics={'sigmoid_output': tf.keras.metrics.binary_accuracy},
        loss_weights={'logits_variance': logit_weight, 'sigmoid_output': sigmoid_weight})
    return model




if False:
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model

    def resnet50(input_tensor):
        base_model = tf.keras.applications.ResNet50(include_top=False, input_tensor=input_tensor)
        # freeze encoder layers to prevent over fitting
        for layer in base_model.layers:
            layer.trainable = False

        output_tensor = Flatten()(base_model.output)
        return output_tensor #Model(inputs=input_tensor, outputs=output_tensor)

    def create_bayesian_model(input_shape, output_classes):
        encoder_input_tensor = Input(shape=input_shape)
        encoder_out = resnet50(encoder_input_tensor)
        input_tensor = Input(shape=encoder_out.shape[1:])
        x = BatchNormalization(name='post_encoder')(encoder_out)
        x = Dropout(0.5)(x)
        x = Dense(500, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(100, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        logits = Dense(output_classes)(x)
        variance_pre = Dense(1)(x)
        variance = Activation('softplus', name='variance')(variance_pre)
        logits_variance = concatenate([logits, variance], name='logits_variance')
        softmax_output = Activation('softmax', name='softmax_output')(logits)

        model = Model(inputs=encoder_input_tensor, outputs=[logits_variance,softmax_output])
        return model

    model = create_bayesian_model([28, 28, 3], 10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3, decay=0.001),
        loss={
        'logits_variance': bayesian_categorical_crossentropy(100, 10),
        'softmax_output': 'categorical_crossentropy'
        },
        metrics={'softmax_output': tf.keras.metrics.categorical_accuracy},
        loss_weights={'logits_variance': .2, 'softmax_output': 1.})

    model.summary()

    # Training parameters
    batch_size = 128
    num_classes = 10
    epochs = 5

    # The data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # x_train = x_train.reshape(-1, 784)
    # x_test = x_test.reshape(-1, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    x_train = tf.tile(tf.expand_dims(x_train, -1), [1, 1, 1, 3])
    x_test = tf.tile(tf.expand_dims(x_test, -1), [1, 1, 1, 3])


    # Train the model
    model.fit(
        x_train, tf.one_hot(y_train, num_classes, dtype='int32'),
        batch_size=batch_size, epochs=epochs, validation_split=0.15)

    # Test the model
    model.evaluate(x_test, tf.one_hot(y_test, num_classes))
