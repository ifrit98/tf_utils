import tensorflow as tf

class ResblockBatchnorm1D(tf.keras.Model):
    def __init__(self, filters, kernel_size, padding='same'):
        super(ResblockBatchnorm1D, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters, kernel_size, padding=padding)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.activations.relu
        self.conv2 = tf.keras.layers.Conv1D(filters, kernel_size, padding=padding)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.layer_add = tf.keras.layers.add

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        bn1   = self.batch_norm1(conv1)
        act   = self.activation(bn1)
        conv2 = self.conv2(act)
        bn2   = self.batch_norm2(conv2)
        add   = self.layer_add([bn2, conv1])
        out   = self.activation(add)
        return out

class ResblockBatchnorm2D(tf.keras.Model):
    def __init__(self, filters, kernel_size, padding='same'):
        super(ResblockBatchnorm2D, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.activations.relu
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.layer_add = tf.keras.layers.add

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        bn1   = self.batch_norm1(conv1)
        act   = self.activation(bn1)
        conv2 = self.conv2(act)
        bn2   = self.batch_norm2(conv2)
        add   = self.layer_add([bn2, conv1])
        out   = self.activation(add)
        return out

class ResblockBatchnormBottleneck1D(tf.keras.Model):
    def __init__(self, filters, kernel_size, padding='same'):
        super(ResblockBatchnormBottleneck1D, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters, 1, padding=padding)
        self.conv2 = tf.keras.layers.Conv1D(filters, kernel_size, padding=padding)
        self.conv3 = tf.keras.layers.Conv1D
        self.activation = tf.keras.activations.relu
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.layer_add = tf.keras.layers.add

    def build(self, input_shape):
        self.conv3 = self.conv3(filters=input_shape[-1], kernel_size=1)
        self.built = True

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        bn1   = self.batch_norm1(conv1)
        act1  = self.activation(bn1)
        conv2 = self.conv2(act1)
        bn2   = self.batch_norm2(conv2)
        act2  = self.activation(bn2)
        conv3 = self.conv3(act2)
        bn3   = self.batch_norm3(conv3)
        add   = self.layer_add([bn3, inputs])
        out   = self.activation(add)
        return out

class ResblockBatchnormBottleneck2D(tf.keras.Model):
    def __init__(self, filters, kernel_size, padding='same'):
        super(ResblockBatchnormBottleneck2D, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 1, padding=padding)
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding)
        self.conv3 = tf.keras.layers.Conv1D
        self.activation = tf.keras.activations.relu
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.layer_add = tf.keras.layers.add

    def build(self, input_shape):
        self.conv3 = self.conv3(filters=input_shape[-1], kernel_size=1)
        self.built = True

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        bn1   = self.batch_norm1(conv1)
        act1  = self.activation(bn1)
        conv2 = self.conv2(act1)
        bn2   = self.batch_norm2(conv2)
        act2  = self.activation(bn2)
        conv3 = self.conv3(act2)
        bn3   = self.batch_norm3(conv3)
        add   = self.layer_add([bn3, inputs])
        out   = self.activation(add)
        return out