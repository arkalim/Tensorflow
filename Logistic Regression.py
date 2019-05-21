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

no_of_epochs = 1000
loss_per_epoch = []

###########################################################################################
# Data Preprocessing 
###########################################################################################

# Now we load the boston house price dataset using tensorflow contrib Datasets
iris_dataset = tf.contrib.learn.datasets.load_dataset('iris')

#Print the data shape
print(iris_dataset.data.shape)

# Separate the data into Train data and labels
# Here we are considering all the features
X_train = iris_dataset.data
Y_train = iris_dataset.target

print(X_train.shape)
print(Y_train.shape)


# Normalize the training data
X_train = Normalize(X_train)

# First index gives the number of samples
n_samples = X_train.shape[0]

# Second index gives the number of features
n_features = X_train.shape[1]

###########################################################################################
# Creating Computational Graph
###########################################################################################

# Placeholder for storing the training data
X = tf.placeholder(tf.float32 , name = 'X', shape = [n_samples , n_features])
Y = tf.placeholder(tf.float32 , name = 'Y')

# Assigning weights and biases randomly
w = tf.Variable(tf.random_normal([n_features , 1] , seed = 12) , name = 'weight')
b = tf.Variable(tf.random_normal([n_samples , 1] ,  seed = 12) , name = 'biases')

# Linear Regression Model for prediction
y_pred = tf.add(tf.matmul(X,w) , b) 
y_pred = tf.sigmoid(y_pred)

# Loss function(MSE)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = y_pred))

# Gradient descent optimiter for minimizing loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

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
        
        # Feeding the data and finding the loss in the current epoch
        # Evaluating the train and loss nodes
        _ , current_loss = sess.run([optimizer , loss] , feed_dict = {X : X_train , Y : Y_train})
                  
        # Append the current loss
        loss_per_epoch.append(current_loss)
        
        #Printing average loss in each epoch
        print('Epoch {0} : Loss {1}'.format(epoch , current_loss))
        
    writer.close()  

    # Getting the value of weight and bias 
    w_trained , b_trained = sess.run([w , b])
    #print('Weights {0}: Bias {1}'.format(w_value, b_value))
       
plt.plot(loss_per_epoch)
plt.show()

