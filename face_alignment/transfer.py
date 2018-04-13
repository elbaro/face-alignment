import tinder
tinder.setup(parse_args=False)
import torch
import tensorflow as tf
import models
import models_tensorflow

x = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='img')
x = models_tensorflow.FAN(x)

torch_net = models.FAN(4)
torch_net.load_state_dict(
    torch.load('3DFAN-4.pth.tar', map_location=lambda storage, loc: storage)
)


# Pytorch [out_channels, in_channels, kernel_size[0] (h), kernel_size[1] (w)]
# TF      [filter_height, filter_width, in_channels, out_channels]

def rgetattr(obj, path):
    for part in path.split('.'):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj

with tf.Session() as sess:
    print('trasnferring ..')
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):

        if v.name.endswith('/kernel:0') or v.name.endswith('/bias:0'):
            # conv2d_57/kernel:0 -> conv2d_57.weight
            # conv2d_57/bias:0 -> conv2d_57.bias
            name = v.name.replace('kernel','weight')
        else:
            # batch_normalization_128/gamma:0 -> bn.weight
            # batch_normalization_128/beta:0 -> bn.bias
            # batch_normalization_128/moving_mean:0 -> bn.running_mean
            # batch_normalization_128/moving_variance:0 -> bn.running_var
            name = v.name.replace('gamma','weight').replace('beta','bias').replace('moving','running').replace('variance','var')

        name = name.replace('/','.').replace(':0','')


        print(v.name, v.get_shape(), name, rgetattr(torch_net, name).size())

        torch_tensor = rgetattr(torch_net, name)
        torch.is_tensor
        if not torch.is_tensor(torch_tensor):
            torch_tensor = torch_tensor.data

        if len(v.get_shape()) == 4:
            v.load(torch_tensor.permute(2, 3, 1, 0).numpy())
        elif len(v.get_shape()) == 1:
            v.load(torch_tensor.numpy())
        else:
            raise Exception("unknown shape")
    print('transfer done.')

    print('saving ..')

    saver = tf.train.Saver()
    saver.save(sess, 'tf/weights.ckpt')

    graph_def = sess.graph.as_graph_def()
    tf.train.write_graph(graph_def, 'graph_def', 'graph_def.pb', False)

    print('saving done.')
