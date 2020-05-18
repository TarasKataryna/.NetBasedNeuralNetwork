import matplotlib.pyplot as plt
import sys as sys


if __name__ == '__main__':
    
    loss = []
    epoch = []

    with open("C:\\Users\\Documents\\Github\\.NetBaseNeuralNetwork\\Data\\loss_results.txt", 'r') as f:
        loss = [float(i) for i in f.readline().split(';')]
        epoch = [float(i) for i in f.readline().split(';')] 
    
    plt.plot(epoch, loss) 
    plt.ylabel('Loss')
    plt.xlabel('Step') 
    plt.show()
    