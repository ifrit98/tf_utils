import tensorflow as tf

# expects that input will be the output of another conv later (e.g. high number of filters)
class SqueezeExcite(tf.keras.Model):
    def __init__(self, filters, reduction=2, downsample=False):
        super(SqueezeExcite, self).__init__()
        reduced = filters // reduction
        self.avg_pool = tf.keras.layers.AveragePooling2D(1)
        self.conv1 = tf.keras.layers.Conv2D(reduced, 1)
        self.relu = tf.keras.activations.relu
        self.conv2 = tf.keras.layers.Conv2D(filters, 1)
        self.sigmoid = tf.keras.activations.sigmoid
        self.mult = tf.keras.layers.multiply
        self.downsample = downsample

    def build(self, input_shape):
        self.built = True

    def call(self, x):
        residual = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        if self.downsample:
            residual = tf.keras.layers.Conv2D(x.shape[-1], 1)(residual)
        return self.mult([x, residual])

class SEResNeXtBottleneck(tf.keras.Model):
    def __init__(self, filters, reduction=16, base_width=4,
                 stride=1, expansion=2, downsample=False):
        super(SEResNeXtBottleneck, self).__init__()
        width = tf.math.floor(filters * (base_width / 64)) * 32
        width = int(width.numpy())
        self.conv1 = tf.keras.layers.SeparableConv2D(
            width, 3, padding='same', depth_multiplier=1)
        self.conv2 = tf.keras.layers.SeparableConv2D(
            width, 3, padding='same', depth_multiplier=1)
        self.conv3 = tf.keras.layers.Conv2D(
            filters * expansion, 1, use_bias=False, padding='same')
        self.se_module = SqueezeExcite(filters * expansion, reduction, downsample)

    def build(self, input_shape):
        self.built = True

    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = self.conv2(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = self.conv3(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self.se_module(x)
        if x.shape[-1] != residual.shape[-1]: # downsample
            residual = tf.keras.layers.Conv2D(x.shape[-1], 1, padding='same')(residual)
        return tf.keras.layers.add([x, residual])


from collections import OrderedDict

class SENet(tf.keras.Model):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):        
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False)),
                ('bn1', tf.keras.layers.BatchNormalization(64)),
                ('relu1', tf.keras.activations.relu),
                ('conv2', tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)),
                ('bn2', tf.keras.layers.BatchNormalization(64)),
                ('relu2', tf.keras.activations.relu),
                ('conv3', tf.keras.layers.Conv2D(inplanes, 3, padding='same', use_bias=False)),
                ('bn3', tf.keras.layers.BatchNormalization(inplanes)),
                ('relu3', tf.keras.activations.relu)
            ]
        else:
            layer0_modules = [
                ('conv1', tf.keras.layers.Conv2D(inplanes, kernel_size=7, strides=2,
                                    padding='same', use_bias=False)),
                ('bn1', tf.keras.layers.BatchNormalization(inplanes)),
                ('relu1', tf.keras.activations.relu),
            ]

        layer0_modules.append(('pool', tf.keras.layers.MaxPool2D(3, strides=2)))
        self.layer0 = tf.keras.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding='same'
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = tf.keras.layers.AvgPooling2D(7, strides=1)
        self.dropout = tf.keras.layers.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = tf.keras.layers.Dense(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential(
                tf.keras.layers.Conv2D(planes * block.expansion,
                          kernel_size=downsample_kernel_size, strides=stride,
                          padding=downsample_padding, use_bias=False),
                tf.keras.layers.BatchNormalization(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return tf.keras.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def call(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
    
def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding='same',
                  num_classes=num_classes)
    return model


# se_resnext50_32x4d()


def make_layer(x, block, filters, no_blocks, reduction, expansion, stride=1):
    shape = x.shape.as_list()
    downsample = None
    inchannels = shape[-1]

    if stride != 1 or inchannels != filters * expansion:
        x = tf.keras.layers.Conv2D(filters * expansion)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    
    x = block(filters)


# TODO: Finish making SEResNeXtBottleneck model



# class Bottleneck(tf.keras.layers.Layer):
#     def call(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out = self.se_module(out) + residual
#         out = self.relu(out)

#         return out


# class SEBottleneck(Bottleneck):
#     """
#     Bottleneck for SENet154.
#     """
#     expansion = 4

#     def __init__(self, inplanes, planes, groups, reduction, stride=1,
#                  downsample=None):
#         super(SEBottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes * 2)
#         self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
#                                stride=stride, padding=1, groups=groups,
#                                bias=False)
#         self.bn2 = nn.BatchNorm2d(planes * 4)
#         self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
#                                bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU()
#         self.se_module = SEModule(planes * 4, reduction=reduction)
#         self.downsample = downsample
#         self.stride = stride


# class SEResNetBottleneck(Bottleneck):
#     """
#     ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
#     implementation and uses `stride=stride` in `conv1` and not in `conv2`
#     (the latter is used in the torchvision implementation of ResNet).
#     """
#     expansion = 4

#     def __init__(self, inplanes, planes, groups, reduction, stride=1,
#                  downsample=None):
#         super(SEResNetBottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
#                                stride=stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
#                                groups=groups, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU()
#         self.se_module = SEModule(planes * 4, reduction=reduction)
#         self.downsample = downsample
#         self.stride = stride
