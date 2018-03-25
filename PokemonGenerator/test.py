import tensorflow as tf

width = 32
height = 32
batch_size = 10
nb_epochs = 15
code_length = 128

graph = tf.Graph()

from keras.datasets import cifar10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
with graph.as_default():
    # Global step
    global_step = tf.Variable(0, trainable=False)

    # Input batch
    input_images = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))

    # Convolutional layer 1
    conv1 = tf.layers.conv2d(inputs=input_images,
                             filters=32,
                             kernel_size=(3, 3),
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.tanh)

    # Convolutional output (flattened)
    conv_output = tf.contrib.layers.flatten(conv1)

    # Code layer
    code_layer = tf.layers.dense(inputs=conv_output,
                                 units=code_length,
                                 activation=tf.nn.tanh)

    # Code output layer
    code_output = tf.layers.dense(inputs=code_layer,
                                  units=(height - 2) * (width - 2) * 3,
                                  activation=tf.nn.tanh)

    # Deconvolution input
    deconv_input = tf.reshape(code_output, (batch_size, height - 2, width - 2, 3))

    # Deconvolution layer 1
    deconv1 = tf.layers.conv2d_transpose(inputs=deconv_input,
                                         filters=4,
                                         kernel_size=(5, 5),
                                         strides=[2,2],
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         activation=tf.sigmoid)

    print deconv1.shape

    # Output batch
    output_images = tf.cast(tf.reshape(deconv1,
                                       (batch_size, height, width, 3)) * 255.0, tf.uint8)

    # Reconstruction L2 loss
    loss = tf.nn.l2_loss(input_images - deconv1)

    # Training operations
    learning_rate = tf.train.exponential_decay(learning_rate=0.0005,
                                               global_step=global_step,
                                               decay_steps=int(X_train.shape[0] / (2 * batch_size)),
                                               decay_rate=0.95,
                                               staircase=True)

    trainer = tf.train.RMSPropOptimizer(learning_rate)
    training_step = trainer.minimize(loss)
