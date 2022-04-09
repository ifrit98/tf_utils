import tensorflow as tf

class PyramidalRecurrentBlock(tf.keras.layers.Layer):
    """ Pyramidal recurrent block
    
    Builds a block of `num_layers` pyramidal recurrent layers, which references https://arxiv.org/pdf/1808.09029.pdf
    
    Params:
        units Number of hidden units for each layer in the block
        cell_type string denoting type of cell to use. default: `LSTM`
        activation activation function for each layer
        ... blah blah TODO

    """
    def __init__(self,
                 units,
                 batch_size,
                 num_layers=3,
                 cell_type='GRU',
                 activation='tanh',
                 projection_activation=True,
                 projection_batchnorm=True,
                 kernel_initializer='glorot_normal',
                 kernel_regularizer=None,
                 recurrent_activation='sigmoid',
                 recurrent_initializer='orthogonal',
                 recurrent_regularizer=None,
                 recurrent_dropout=0.0,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 trainable=True):
        super(PyramidalRecurrentBlock, self).__init__()
        self.units = units
        self.num_layers = num_layers
        self.cell_type = str(cell_type).upper()
        self.activation = tf.keras.activations.get(activation)
        self.projection_activation =  projection_activation
        self.projection_batchnorm =  projection_batchnorm
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.recurrent_regularizer = tf.keras.regularizers.get(recurrent_regularizer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.recurrent_dropout = recurrent_dropout
        self.trainable = trainable
        self.layers = []
        self.batch_size = batch_size
        self.useable_layers = {
                "LSTM": tf.keras.layers.LSTM,
                "GRU": tf.keras.layers.GRU,
                "RNN": tf.keras.layers.SimpleRNN
            }

    def build(self, input_shape):
        self.cell = self.useable_layers.get(self.cell_type, tf.keras.layers.GRU)

        for i in range(self.num_layers):
            self.layers.append(
                tf.keras.layers.Bidirectional(
                layer=self.cell(
                    units=self.units,
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True,
                    return_state=True),
                backward_layer=self.cell(
                    units=self.units,
                    recurrent_dropout=self.recurrent_dropout,
                    go_backwards=True,
                    return_sequences=True,
                    return_state=True),
                merge_mode='concat',
                name='bidirectional_layer_{}'.format(i)
                )
            )

    def call(self, x, mask=None):
        output = x
        i = 0
        for layer in self.layers:
            out = layer(output)
            # print("\nOutput:", out)
            if self.cell_type == "LSTM":
                output, _, context_forward, _, context_backward = out
            else:
                output, context_forward, context_backward = out

            output = tf.concat(output, -1)
            state = tf.concat([context_forward, context_backward], -1)

            if i > 0:
                output = self.pad_sequence(output)
                # print("\nOUTPUT AFTER PAD:", output)
                output = tf.keras.layers.Dense(self.units)(output)
            if self.projection_batchnorm:
                output = tf.keras.layers.BatchNormalization()(output)
            if self.projection_activation:
                output = tf.keras.activations.relu(output)
            # print("\noutput after projection:", output)
            i += 1
        
        return (output, state)

    def pad_sequence(self, output):
        shp = output.shape #tf.shape(output) #output.get_shape()
        # print("\nOUTPUT INTO PAD SEQ:", output)
        # print("TF.SHAPE(OUTPUT):", shp)
        # print("OUTPUT.SHAPE", output.shape)
        batch = self.batch_size
        sequence_length = shp[1]
        units = shp[2]
        # print("batch", batch)
        # print("seq_len", shp[1])
        # print("units\n", shp[2])

        floormod = tf.math.floormod(sequence_length, 2)
        padding = [
                    [0, 0],
                    [0, floormod],
                    [0, 0]
                    ]
        new_units = tf.math.multiply(units, 2)
        new_sequence_length = tf.math.floordiv(sequence_length, 2) + floormod
        new_shape = tf.stack([batch, new_sequence_length, new_units]) 
        output = tf.pad(output, padding)

        return tf.reshape(output, new_shape)

    # def compute_output_shape(self, input_shape):
    #     output_shape = list(
    #         input_shape[1],
    #         int(input_shape[2] // 2 ^ (self.num_layers - 1)),
    #         input_shape[3]
    #     )
    #     return output_shape


PRU = PyramidalRecurrentBlock

if False:

    from tf_dataset import *
    from tf_preprocess import *
    import tensorflow as tf 

    ds = signal_dataset(construct_metadata('../tf_dataset/data'), use_soundfile=True)
    ds = dataset_signal_slice_windows(ds, 128000)
    ds = dataset_signal_normalize_gain(ds)
    ds = dataset_signal_apply_analytic(ds)
    ds = dataset_prepare_signal_for_ml(ds)
    ds = dataset_set_shapes(ds)
    ds = dataset_compact(ds, 'signal_features')
    ds = ds.shuffle(10)
    ds = ds.batch(8)
    ds = ds.prefetch(1)
    for x in ds: break

    l = tf.keras.layers
    batched_x_shape = x[0].shape
    inputs = l.Input(batch_shape=batched_x_shape, dtype='float32')
    bn   = l.BatchNormalization()(inputs)
    base = PRU(32, batch_size=batched_x_shape[0])(bn)
    bn   = l.BatchNormalization()(base[0])
    base = PRU(64, batch_size=batched_x_shape[0])(bn)
    bn   = l.BatchNormalization()(base[0])
    down = l.GlobalMaxPooling1D()(bn)
    down = l.Dense(64, activation='relu')(down)
    down = l.Dense(32, activation='relu')(down)
    down = l.Dense(8, activation='relu')(down)
    out  = l.Dense(1, activation='sigmoid')(down)
    optimizer = tf.keras.optimizers.get('adam')
    model = tf.keras.Model(inputs=inputs, outputs=out)
    loss_fn = 'binary_crossentropy'  
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    print(model.summary())


    model.fit(
        ds,
        epochs=5,
        steps_per_epoch=5
    )

    # Track output shape changes across LSTM/GRU cells (No context vector in GRU)
    lstm = tf.keras.layers.LSTM(
        4, return_sequences=True, return_state=True)
    rev_lstm = tf.keras.layers.LSTM(
        4, return_sequences=True, return_state=True, go_backwards=True)
    out = lstm(x)
    print(len(out))
    whole_seq_output, final_memory_state, final_carry_state = lstm(x)
    print(whole_seq_output.shape)
    print(final_memory_state.shape)
    print(final_carry_state.shape)

    blstm = tf.keras.layers.Bidirectional(
        layer=lstm, merge_mode='concat', backward_layer=rev_lstm)
    bout = blstm(x)
    print(len(bout))
