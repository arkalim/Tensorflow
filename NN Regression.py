import tensorflow as tf
import tensorflow.contrib.layers as layers
from sklearn import datasets
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt

test_example_number = 100
max_epoch = 10
n_hidden = 500
prediction = []

##########################################################################################
# Data Preprocessing
##########################################################################################
boston_dataset = datasets.load_boston()

originial_X_train = boston_dataset.data
original_Y_train = boston_dataset.target

X_train, X_test, y_train, y_test = train_test_split(originial_X_train , original_Y_train , test_size=0.3, random_state=0)

print(" Training data size")
print(X_train.shape , y_train.shape)
print(" Testing data size")
print(X_test.shape , y_test.shape)

# Target has to be rescaled as fit_transform only takes 2d array
y_train_reshape = np.reshape(y_train , (354,1))
y_test_reshape = np.reshape(y_test , (152,1))

# Normalize the train data
X_train = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_train)
y_train = MinMaxScaler(feature_range=(0, 1)).fit_transform(y_train_reshape)

# Normalise the test data
# If we use tanh activation function then its better to use feature_range=(-1, 1)
# when we did linear regression there was no activation function hence only the inputs were scaled
X_test =  MinMaxScaler(feature_range=(0, 1)).fit_transform(X_test)
y_test =  MinMaxScaler(feature_range=(0, 1)).fit_transform(y_test_reshape)

# Number of samples in the training set 
n_samples = len(X_train)

# Number of samples in the test set 
num_test_samples = len(X_test)

###########################################################################################

#Defining the Neural Network
def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, n_hidden, activation_fn = tf.nn.relu)
    fc2 = layers.fully_connected(fc1, n_hidden, activation_fn = tf.nn.relu)
    fc3 = layers.fully_connected(fc2, n_hidden, activation_fn = tf.nn.relu)
    fc4 = layers.fully_connected(fc3, n_hidden, activation_fn = tf.nn.relu)
    fc5 = layers.fully_connected(fc4, n_hidden, activation_fn = tf.nn.relu)
    fc6 = layers.fully_connected(fc5, n_hidden, activation_fn = tf.nn.relu)
    fc7 = layers.fully_connected(fc6, n_hidden, activation_fn = tf.nn.relu)
    fc8 = layers.fully_connected(fc7, n_hidden, activation_fn = tf.nn.relu)
    fc9 = layers.fully_connected(fc8, n_hidden, activation_fn = tf.nn.relu)
    out = layers.fully_connected(fc9, 1, activation_fn = tf.tanh)
    return out

# Defining the input and output placeholders
X = tf.placeholder(tf.float32, name='X' , shape = [None , 13])
Y = tf.placeholder(tf.float32, name='Y')

# Creating the NN model
y_hat = multilayer_perceptron(X)

# Defining the loss function(MSE)
loss =  tf.reduce_mean(tf.cast(tf.square(Y - y_hat), tf.float32))

# Defining the optimisation funtion(Adams)
optimizer = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(loss)

# Defining the variable initialisation function
init = tf.global_variables_initializer()

with tf.Session() as sess:
        
    # Initialize variables
    sess.run(init)

    # Train the model
    for epoch in range(max_epoch):
        
        _, current_loss = sess.run([optimizer, loss], feed_dict={X: X_train, Y: y_train})
        print('Epoch {0}: Loss {1}'.format(epoch, current_loss))
        
    print("Model is trained")
    
    # Calculate accuracy
    
    for sample in range(num_test_samples):
    
        # Reshaping the input for prediction 
        reshaped_input = np.reshape(X_test[sample] , (1 , 13))
        
        # Finding the prediction and loss 
        _, current_pred_loss, y_pred = sess.run([optimizer, loss, y_hat], feed_dict={X: reshaped_input, Y: y_test})
        prediction.append(y_pred[0][0])

###########################################################################################
# Plotting the result
############################################################################################    

# Finding the order of the y_test
order = np.argsort(y_test, axis=0)

# Sort y_test
sorted_y_test = np.sort(y_test,axis = 0)

# Sort prediction in the same order as that of y_test
sorted_prediction = [x for _,x in sorted(zip(order,prediction))]

# Plot the regression chart
plt.title("Regression Chart")
plt.plot(sorted_y_test, 'c', label='Real Data')
plt.plot(sorted_prediction,'m', label='Predicted Data')
plt.legend()

plt.show()