# Loading the required packages
import tensorflow as tf
import matplotlib.pyplot as plt 
import tensorflow.contrib.layers as layers
from sklearn.preprocessing import MinMaxScaler

# Empty lists for storing the loss and accuracy history
train_loss = []
train_acc = []
validation_acc = []
validation_loss = []

# Parameters
learning_rate = 0.001
training_iters = 500
batch_size = 1024
display_step = 10
no_hidden_units = 1024

###########################################################################################
# Data Preprocessing 
###########################################################################################

# Now we load the mnist dataset
# each handwritten digit is of the size 28*28 i.e. 784 pixels grayscale image
from tensorflow.examples.tutorials.mnist import input_data

# we use one hot encoding for labels
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 55000 examples with grayscale image of 784 pixels
print(mnist.train.images.shape)
# the labels are one hot encoded
print(mnist.train.labels.shape)
# 10000 test examples
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

# Fetch the validation data and normalize it
# Training data is normalized after fetching the batch of training data
X_valid = mnist.test.images
X_valid = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_valid)
Y_valid = mnist.test.labels

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.85 # Dropout, probability to keep units

###########################################################################################
# Creating Computational Graph
###########################################################################################


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return x

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def model(x, weights, biases, dropout):
    # reshape the input picture
    # -1 represents that the number of images is unknown
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # First convolution layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Output size is 28*28*32
    # Max Pooling used for downsampling
    # output is 14*14*32
    conv1 = maxpool2d(conv1, k=2)

    # Second convolution layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Output is (14,14,64)
    # Max Pooling used for downsampling
    conv2 = maxpool2d(conv2, k=2)
    # Output is (7,7,64)

    # Flatten conv2 output to match the input of fully connected layer 
    # To get the shape as a list of ints, do tensor.get_shape().as_list()
    # the shape is -1,7*7*64
    flatten = tf.reshape(conv2, [-1,7*7*64])

    #We have two fully connected layers with relu activation func
    fc1 = layers.fully_connected(flatten, no_hidden_units, activation_fn = tf.nn.relu )
    fc2 = layers.fully_connected(fc1, no_hidden_units, activation_fn = tf.nn.relu)
    
    # We are applying softmax activation in the output layer 
    out = layers.fully_connected(fc2, n_classes, activation_fn=None)
    return out

weights = {
    # 5x5 conv, 1 input, and 32 outputs( or 32 filters)
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, and 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
}

# Defining the features and labels placeholders
X = tf.placeholder(tf.float32 ,shape = [None , 784] , name='X')
Y = tf.placeholder(tf.float32,shape = [None , 10],name='Y')
keep_prob = tf.placeholder(tf.float32)

# Create the CNN model
# keep_prob is the probability to keep a node during training
pred = model(X, weights, biases, keep_prob)

# Defining the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

# Defining the optimization function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# The prediction is correct when Y equals pred
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

# Defining the accuracy
# Type casting the prediction to float value and averaging over the entire set
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Defining the variable initialisation function
init = tf.global_variables_initializer()

############################################################################################
# Training the model 
############################################################################################

with tf.Session() as sess:
    
    # Initialise the variables
    sess.run(init)

    for step in range(training_iters):
        
        # Fetching the next batch of data
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        # Normalising the data
        batch_x = MinMaxScaler(feature_range=(0, 1)).fit_transform(batch_x)
        
        # Train our model on the batch of data
        sess.run(optimizer, feed_dict={X : batch_x , Y : batch_y , keep_prob : dropout})
        
        # Displaying the loss and accuracy
        if step % display_step == 0:
            
            # Evaluate the train loss and accuracy with no dropout
            loss_train, acc_train = sess.run([loss , accuracy], feed_dict={X: batch_x,Y: batch_y,keep_prob: 1.0})
    
            # Evaluate the test accuracy with no dropout
            loss_valid, acc_valid = sess.run([loss , accuracy], feed_dict={X: X_valid, Y: Y_valid,keep_prob: 1.0})
            print ("Epoch " + str(step) + ", Train Loss= " + "{:.2f}".format(loss_train) + ", Training Accuracy= " + "{:.2f}".format(acc_train) + ", Validation Loss= " + "{:.2f}".format(loss_valid)+ ", Validation Accuracy:" + "{:.2f}".format(acc_valid))
    
            # Append the data for plotting
            train_loss.append(loss_train)
            train_acc.append(acc_train)
            validation_acc.append(acc_valid)
            validation_loss.append(loss_valid)
            
###########################################################################################
# Plotting the loss and accuracy
############################################################################################
plt.subplot(1,2,1)
plt.title("Loss Curve")
plt.plot(train_loss, 'r', label='Training Loss')
plt.plot(validation_loss,  'c', label='Validation Loss')
plt.legend()

plt.subplot(1,2,2)
plt.title("Accuracy Curve")
plt.plot(train_acc, 'r', label='Training Loss')
plt.plot(validation_acc, 'c', label='Validation Loss')
plt.legend()

plt.show()            
