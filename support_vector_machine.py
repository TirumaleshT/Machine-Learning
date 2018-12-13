import numpy as np

def train():
	data = load_data()
	learning_rate = 0.1
	epochs = 10000
	X = data[:,:2]
	Y = data[:, 2]
	#print(Y)
	svm_sgd(data, learning_rate, epochs)

#load data
def load_data():
	data = np.genfromtxt('pulse_classify.csv', delimiter = ',')
	return data
	
def svm_sgd(data, learning_rate, epochs):
	X = data[:,:2]
	Y = data[:, 2]
	w = np.zeros(len(X[0]))
	for epoch in range(1, epochs):
		for i, x in enumerate(X):
			#misclassified
			if (Y[i]*np.dot(X[i], w)) < 1:
				w += learning_rate * ((X[i]*Y[i]) - 2*(1/epoch)*w)
			else:
			#correct classification
				w += learning_rate * (- 2*(1/epoch)*w)
				
if __name__ == '__main__':
	import os
	os.chdir('C:\\Users\\Tirumalesh\\Downloads\\Datasets')
	train()