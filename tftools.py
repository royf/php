import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def linear(s, output_size):
    input_size = int(s.shape[1])
    W_fc = weight_variable([input_size, output_size])
    b_fc = bias_variable([output_size])
    return tf.matmul(s, W_fc) + b_fc


def mlp(s, hidden_sizes, output_size):
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    layer = s
    shape = s.shape.as_list()[:-1]
    size1 = int(s.shape[-1])
    for size2 in hidden_sizes:
        W = weight_variable([size1, size2])
        b = bias_variable([size2])
        layer = tf.tensordot(layer, W, 1)
        layer.set_shape(shape + [size2])
        layer = tf.nn.relu(layer + b)
        size1 = size2
    W = weight_variable([size1, output_size])
    b = bias_variable([output_size])
    out = tf.tensordot(layer, W, 1)
    out.set_shape(shape + [output_size])
    return out + b


def multi_mlp(s, hidden_size, output_size, num_nets, name=None):
    input_size = int(s.shape[-1])
    with tf.variable_scope(name, default_name='multi_mlp'):
        W_fc1 = tf.get_variable(
            'W_fc1', [num_nets, input_size, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_fc1 = tf.get_variable(
            'b_fc1', [num_nets, hidden_size],
            initializer=tf.zeros_initializer)

        W_fc2 = tf.get_variable(
            'W_fc2', [num_nets, hidden_size, output_size],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_fc2 = tf.get_variable(
            'b_fc2', [num_nets, output_size],
            initializer=tf.zeros_initializer)

        net = tf.tensordot(s, W_fc1, axes=[[1], [1]]) + b_fc1  # [batch_size, num_nets, hidden_size]
        net = tf.nn.relu(net)
        net = tf.transpose(net, [1, 0, 2])  # [num_nets, batch_size, hidden_size]
        net = tf.matmul(net, W_fc2) + tf.expand_dims(b_fc2, axis=1)  # [num_nets, batch_size, output_size]
        return net


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
