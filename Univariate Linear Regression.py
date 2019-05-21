# Loading the required packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Function to normalize the input data(array form)
def Normalize(x):
    mean = np.mean(x)
    standard_deviation = np.std(x)
    X = (x - mean)/standard_deviation
    return X

no_of_epochs = 10
total = []

###########################################################################################
# Data Preprocessing 
###########################################################################################

# Now we load the boston house price dataset using tensorflow contrib Datasets
boston_dataset = tf.contrib.learn.datasets.load_dataset('boston')

#Print the data shape
print(boston_dataset.data.shape)

# Separate the data into X_train and Y_train
# By 5 means out of 13 columns(features) we are only taking 5th one into consideration
# Selecting 5th column
X_train , Y_train = boston_dataset.data[:,5] , boston_dataset.target

#Normalise the training data
X_train = Normalize(X_train)
n_samples = len(X_train)

###########################################################################################
# Creating Computational Graph
###########################################################################################

# Placeholder for storing the training data
X = tf.placeholder(tf.float32 , name = 'X')
Y = tf.placeholder(tf.float32 , name = 'Y')

# Assigning weights and biases to 0
w = tf.Variable(0.0 , name = 'weight')
b = tf.Variable(0.0 , name = 'biases')

# Linear Regression Model for prediction
y_prediction = X*w + b

# Loss function(MSE)
loss = tf.reduce_mean(tf.square( Y - y_prediction, name = 'loss'))

# Gradient descent optimiter for minimizing loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# Initializing variables
init = tf.global_variables_initializer()

############################################################################################
# Training the model 
############################################################################################


with tf.Session() as sess:
    
    # initializing the variables
    sess.run(init)
    
    # for visualization in tensorboard
    writer = tf.summary.FileWriter('graphs' , sess.graph)
    for epoch in range(no_of_epochs):
        total_loss = 0
        
        # Feeding each data point into the optimizer using Feed Dictionary
        for x,y in zip(X_train , Y_train):
            
            # Feeding the data and finding the loss in the current epoch
            # Evaluating the train and loss nodes
            _ , current_loss = sess.run([train , loss] , feed_dict = {X : x , Y : y})
            
            #Total loss = sum of each loss
            total_loss += current_loss
        
        # Appending the average loss in each epoch
        total.append(total_loss/n_samples)
        
        #Printing average loss in each epoch
        print('Epoch {0} : Loss {1}'.format(epoch , total_loss/n_samples))
    writer.close()
    
    # Getting the value of weight and bias 
    w_value , b_value = sess.run([w , b])
    print('Weights {0}: Bias {1}'.format(w_value, b_value))

#Finding the prediction after the model is trained
Y_pred = X_train*w_value + b_value

###########################################################################################
# Plotting the result
############################################################################################
plt.subplot(1,2,1)
plt.title("Regression Chart")
plt.plot(X_train, Y_train, 'go', label='Real Data')
plt.plot(X_train,Y_pred,  'r', label='Predicted Data')


plt.subplot(1,2,2)
plt.title("Loss Curve")
plt.plot(total)

plt.show()