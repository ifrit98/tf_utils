
import tensorflow as tf
K = tf.keras.backend

class GatedConvBlock(tf.keras.layers.Wrapper):
    def __init__(self, conv_layer, conv_num=3,
                 gate_activation='sigmoid', **kwargs):

        super(GatedConvBlock, self).__init__(conv_layer, **kwargs)
        self.conv_num = conv_num
        self.gate_activation = tf.keras.activations.get(gate_activation)
        self.conv_layers = []
        self.input_spec = conv_layer.input_spec 
        self.rank = conv_layer.rank
        self.filters = conv_layer.filters//2
        if conv_layer.padding != 'same':
            raise ValueError("The padding mode of this layer must be `same`"
                            ", But found `{}`".format(conv_layer.padding))
        
        config = conv_layer.get_config()
        if '1d' in config['name']:
            conv_constructor = tf.keras.layers.Conv1D
        elif '2d' in config['name']:
            conv_constructor = tf.keras.layers.Conv2D
        elif '3d' in config['name']:
            conv_constructor = tf.keras.layers.Conv3D
        else:
            raise ValueError("Cannot have greater than rank 3 input")

        for i in range(self.conv_num):
            config['name'] = 'GatedConvBlock_{}_{}'.format(conv_layer.name, i)
            new_conv_layer = conv_constructor(
                filters=config['filters'], 
                kernel_size=config['kernel_size'], 
                padding='same')
            new_conv_layer.from_config(config)
            self.conv_layers.append(new_conv_layer)

    def build(self, input_shape):
        if self.conv_layers[0].filters != input_shape[-1] * 2:
            raise ValueError("For efficiency, the sub-conv-layer filters must be twice\
                that of input_shape[-1].\nFound filters={}, input_shape[-1]={}".format(
                    self.conv_layers[0].filters, input_shape[-1]))

        input_shape_current = input_shape
        for layer in self.conv_layers:
            with K.name_scope(layer.name):
                layer.build(input_shape_current)                
            input_shape_current = input_shape
        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape_current = input_shape
        for layer in self.conv_layers:
            input_shape_current = layer.compute_output_shape(input_shape_current)
            output_shape = list(input_shape_current)
            output_shape[-1] = int(output_shape[-1]/2)
            input_shape_current = output_shape   
        return tuple(input_shape_current)

    def half_slice(self, x):
        ndim = self.rank +2
        if ndim ==3:
            linear_output = x[:,:,:self.filters]
            gated_output = x[:,:,self.filters:]
        elif ndim ==4:
            linear_output = x[:,:,:,:self.filters]
            gated_output = x[:,:,:,self.filters:]
        elif ndim ==5:
            linear_output = x[:,:,:,:,:self.filters]
            gated_output = x[:,:,:,:,self.filters:]
        else:
            raise ValueError("This class only support for 1D, 2D, 3D conv, but\
                input ndim={}".format(ndim))
        return linear_output, gated_output 
 
    def call(self, inputs):
        input_current = inputs  
        for i, layer in enumerate(self.conv_layers):
            output_current = layer(inputs= input_current)     
            linear_output, gated_output = self.half_slice(output_current)
            input_current = linear_output * self.gate_activation(gated_output)
            input_current._keras_shape = K.int_shape(linear_output) 
        
        output = input_current + inputs

        return output 

    def get_weights(self):
        weights = None 
        for layer in self.conv_layers:
            weights += layer.get_weights()
        return weights

    def set_weights(self, weights):
        for layer in self.conv_layers:
            layer.set_weights(weights)
        pass

    @property
    def trainable_weights(self):
        weights = []
        for layer in self.conv_layers:
            if hasattr(layer, 'trainable_weights'):
                weights += layer.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        for layer in self.conv_layers:
            if hasattr(layer, 'non_trainable_weights'):
                weights += layer.non_trainable_weights
        return weights

    @property
    def updates(self):
        updates_ = []
        for layer in self.conv_layers:
            if hasattr(layer, 'updates'):
                updates_ += layer.upates
        return updates_

    @property
    def losses(self):
        losses_ = []
        for layer in self.conv_layers:
            if hasattr(layer, 'losses'):
                losses_ += layer.losses
        return losses_

    @property
    def constraints(self):
        constraints_ = {}
        for layer in self.conv_layers:
            if hasattr(layer, 'constraints'):
                constraints_.update(layer.constraints)
        return constraints_ 

GLU = GatedConvBlock