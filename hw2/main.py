from Layer import Layer
from ANN import ANN
from ANNConfig import ANNConfig
from random import shuffle
from typing import Tuple
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def loadData(filename):
    D = np.genfromtxt(fname=filename)
    D = D[D[:, 0].argsort()]
    X = np.array([D[:, 0].T]).T
    Y = np.array([D[:, 1].T]).T
    return X, Y

#######################################################

#load data
X_train, Y_train = loadData(sys.argv[2])
X_test, Y_test = loadData(sys.argv[3])

#create network
ann = ANN(1111)
ann.loadConfig(ANNConfig(sys.argv[1]))

#normalize if needed
if ann.normalizeInput:
    X_train, max_x_train = ann.normalizeVector(X_train)
    Y_train, max_y_train = ann.normalizeVector(Y_train)
    X_test, max_x_test = ann.normalizeVector(X_test)
    Y_test, max_y_test = ann.normalizeVector(Y_test)

else:
    max_x_train = 1.0
    max_y_train = 1.0
    max_x_test = 1.0
    max_y_test = 1.0


#


trainResult = ann.train(X_train, Y_train)
mse= trainResult[0]
# mse *= max_y_train
mseIndex = trainResult[1]

print("Final loss:", trainResult[0][-1], "in", trainResult[2], "iterations")
print("Loss mean:", np.mean(trainResult[0]), "\nLoss stdev:", np.std(trainResult[0]))


# predict for train data
start = -1.0
end = +1.0
data_size = 1000

x_pred_train = np.array([[x] for x in np.arange(start, end, (end - start) / data_size)])
y_pred_train = ann.predict([x_pred_train])[0]


x_pred_test = np.array([[x] for x in np.arange(start, end, (end - start) / data_size)])
y_pred_test = ann.predict([x_pred_test])[0]



##################################################################

plt.figure()
plt.scatter(X_train*max_x_train, Y_train*max_y_train, color="green", label="Real (x,y)")
plt.plot(x_pred_train*max_x_train, y_pred_train*max_y_train, color="blue", label="Predicted (x,y)")
plt.title("Regression Results for Training Data (" +sys.argv[2]+")")
plt.legend(loc='upper left', frameon=True, facecolor='white')
plt.ylabel('')
plt.xlabel('')
plt.savefig("figs/"+os.path.basename(sys.argv[1])+"_1_"+os.path.basename(sys.argv[2])+"_"+os.path.basename(sys.argv[3])+".png", bbox_inches='tight')


plt.figure()
plt.scatter(X_test*max_x_test, Y_test * max_y_test, color="green", label="Real (x,y)")
plt.plot(x_pred_test*max_x_test, y_pred_test * max_y_test, color="blue", label="Predicted (x,y)")
plt.title("Regression Results for Test Data (" +sys.argv[3]+")")
plt.legend(loc='upper left', frameon=True, facecolor='white')
plt.ylabel('')
plt.xlabel('')
plt.savefig("figs/"+os.path.basename(sys.argv[1])+"_2_"+os.path.basename(sys.argv[2])+"_"+os.path.basename(sys.argv[3])+".png", bbox_inches='tight')


plt.figure()
plt.plot(list(range(len(mse))), mse, label="Mean square error", color="black")
plt.scatter(len(mse), mse[-1], label="Iteration "+str(len(mse)), color="red", marker='|')
plt.title("Loss per Epoch During Training")
plt.legend(loc='upper left', frameon=True, facecolor='white')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.savefig("figs/"+os.path.basename(sys.argv[1])+"_3_"+os.path.basename(sys.argv[2])+"_"+os.path.basename(sys.argv[3])+".png", bbox_inches='tight')

# plt.ioff()
# plt.show()
