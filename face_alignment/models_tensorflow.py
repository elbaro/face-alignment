import tensorflow as tf
import math

conv2d = tf.layers.conv2d
relu = tf.nn.relu
max_pool = tf.contrib.layers.max_pool2d
avg_pool2d = tf.layers.average_pooling2d
bn = tf.layers.batch_normalization
upsample = tf.layers.conv2d_transpose


def conv3x3(x, out_planes, name):
    "3x3 convolution with padding"
    return conv2d(x, out_planes, kernel_size=3, padding='same', use_bias=False, name=name)


def ConvBlock(x, out_planes, name):
    in_planes = x.get_shape().as_list()[-1]
    residual = x

    out1 = bn(x,training=False,name=name+'.bn1', epsilon=1e-5)
    out1 = relu(out1)
    out1 = conv3x3(out1, out_planes // 2, name=name+'.conv1')

    out2 = bn(out1,training=False,name=name+'.bn2', epsilon=1e-5)
    out2 = relu(out2)
    out2 = conv3x3(out2, out_planes // 4, name=name+'.conv2')

    out3 = bn(out2,training=False,name=name+'.bn3', epsilon=1e-5)
    out3 = relu(out3)
    out3 = conv3x3(out3, out_planes // 4, name=name+'.conv3')

    out3 = tf.concat([out1, out2, out3], 3)

    if in_planes != out_planes:
        residual = relu(bn(residual,training=False,name=name+'.downsample.0', epsilon=1e-5))
        residual = conv2d(residual, out_planes, kernel_size=1, strides=1, use_bias=False, name=name+'.downsample.2')

    out3 += residual

    return out3


def HourGlass(x, depth, features, name):
    def _forward(level, inp): # level >= 0
        # Upper branch
        up1 = inp
        up1 = ConvBlock(up1, 256, name=name+'.b1_'+str(level))

        # Lower branch
        low1 = avg_pool2d(inp, 2, strides=2)
        low1 = ConvBlock(low1, 256, name=name+'.b2_'+str(level))

        if level > 1:
            low2 = _forward(level - 1, low1)
        else:
            low2 = low1
            low2 = ConvBlock(low2, 256, name=name+'.b2_plus_'+str(level))

        low3 = low2
        low3 = ConvBlock(low3, 256, name=name+'.b3_'+str(level))

        s = tf.shape(low3)
        h, w = s[1], s[2]
        up2 = tf.image.resize_images(low3, [h*2, w*2], tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

        return up1 + up2

    return _forward(depth, x)


def FAN(x):
    # x = bn(tf.zeros((1,128,128,64)), training=False, name='bn1')
    x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], mode='CONSTANT')
    x = relu(bn(conv2d(x, 64, kernel_size=7, strides=2, padding='valid', name='conv1'), training=False, name='bn1', epsilon=1e-5))
    # x = conv2d(x, 64, kernel_size=7, strides=2, padding='valid', name='conv1')
    # return [x]

    x = avg_pool2d(ConvBlock(x, 128, name='conv2'), 2, strides=2)
    x = ConvBlock(x, 128, name='conv3')
    x = ConvBlock(x, 256, name='conv4')

    previous = x

    outputs = []
    for i in range(4):
        hg = HourGlass(previous,4,256,name='m%d'%i)  # m

        ll = hg
        ll = ConvBlock(ll,256,name='top_m_%d'%i) # top_m

        ll = relu(bn(conv2d(ll, 256, kernel_size=1,name='conv_last%d'%i),name='bn_end%d'%i, epsilon=1e-5)) # bn_end + conv_last

        # Predict heatmaps
        tmp_out = conv2d(ll, 68, kernel_size=1, name='l%d'%i) # l
        outputs.append(tmp_out)

        if i < 3: # not last?
            ll = conv2d(ll, 256, kernel_size=1,name='bl%d'%i) # bl
            tmp_out_ = conv2d(tmp_out,256, kernel_size=1,name='al%d'%i) # al
            previous = previous + ll + tmp_out_

    return outputs
