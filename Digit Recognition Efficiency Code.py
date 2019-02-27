import numpy as np
import matplotlib.pyplot
import scipy.special
import random
class NeuralNetwork:
    
    #parameters of the NeuralNetwork,self is the instance reference
    def __init__(self,inputNodes,hiddenNodes,outputNodes,learningRate):
        
        self.inputNodes=inputNodes
        self.hiddenNodes=hiddenNodes
        self.outputNodes=outputNodes
        self.lr=learningRate
        #Initializing the weights from Input to Hidden Layer
        #The Matrices are wih and who.
        self.wih=np.random.normal(0.0,pow(self.inputNodes,-0.5),(self.hiddenNodes,self.inputNodes))
        self.who=np.random.normal(0.0,pow(self.inputNodes,-0.5),(self.outputNodes,self.hiddenNodes))
        #Implimenting the Sigmoid Function
        self.activation_function=lambda x:scipy.special.expit(x)
        pass
    
    #Training the NeuralNetwork,Parameter[input_list,target_List]
    #Here the Inputs are a normal 1D List
    def train(self,inputs_list,targets_list):

        #We Have to covert the 1D list to 2D Column matrix of the order: 784 by 1
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #calculate the summation of signals into hidden layer
        #X=Weights.Inputs
        hidden_inputs = np.dot(self.wih, inputs)
        #calculaing the Hidden Output
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate the summation of signals into output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        #calculaing the Final Output
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass
      # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

inputNodes=784
outputNodes=10
hiddenNodes=100
learningRate=0.3

#Creating an Instance of NeuralNetwor.
n=NeuralNetwork(inputNodes,hiddenNodes,outputNodes,learningRate)

training_data_file = open("train_60000.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 1

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(outputNodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass
# load the mnist test data CSV file into a list
test_data_file = open("test_10000.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass
# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print ("performance = ", (scorecard_array.sum() / scorecard_array.size)*100,"% Accurate")
