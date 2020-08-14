# Name: Rachael Hawthorne
# Class: Data Mining
# Date: September 23rd, 2019


import numpy as np
import pandas as pd
import csv

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.name = None
        self.splitPoint = None


class DecisionTree:
    def __init__(self):
        self.root = None

    def grow(self, data):
        
        if stop(data) == True:
            leaf = Node()
            leaf.name = str(int(data.mode()['Class'][0])) #assign most likely class
            return leaf
        else:
            print('(Elevator music)...')
            root = Node()
            splitX, splitY, split, attr = best_split(data) #obtain all the split data

            root.name = attr
            root.splitPoint = split

            #recurse on left child
            childX = self.grow(splitX)
            root.left = childX
            #recurse on right child
            childY = self.grow(splitY)
            root.right = childY

        self.root = root
        return self.root

    def getRoot(self):
        return self.root
    
    def printTree(self):
      queue = []
      queue.append((self.root, 0)) #(node, level)
      prev_level = 0
      while (len(queue) > 0):
            node, level = queue.pop(0)
            if level != prev_level:
                print()
            print(node.name, sep='\t', end=' ')
            if (node.left != None):
                queue.append((node.left, level + 1))
            if (node.right != None):
                queue.append((node.right, level + 1))
            prev_level = level


def importData():
    data = pd.read_csv('wine_data.csv', header= None)
    return data


def splitData(dataSet):
    valData = dataSet[::5] #pull out every 5th row for the validation set
    trainData = dataSet.drop(dataSet.iloc[::5].index, 0) #drop every 5th row in the original set for the train set

    return trainData, valData


def best_split(data):
    best_column = None
    best_gini = 2 #started with 2 because ginis can never be over one so it will always be overwritten by the first gini
    best_median = 0

    for column in data:
        if column != 'Class':
            sortedDF = data.sort_values(column, axis=0) #sort data in column
            tempArr = [] #empty list for medians
            prev = 0
            for row in sortedDF.iterrows(): 
                curr = row[1][column] #current row value
                if prev != 0: 
                    median = (curr + prev)/2 #median
                    tempArr.append(median) #add median to list
                prev = curr
            medians = np.array(tempArr) #array of medians for current column
            
            for value in np.nditer(medians):
                splitX, splitY = split(sortedDF, value, column) #split into two arrays by each median
                splitGini = GiniSplit(data, splitX, splitY)

                if splitGini < best_gini:
                    best_gini = splitGini
                    best_median = value
                    best_column = column
    
    best_X, best_Y = split(data, best_median, best_column)

    return best_X, best_Y, best_median, best_column


def split(data, median, column):
    X = pd.DataFrame(columns = ['Class','Alcohol', 'Malic Acid', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols', 'Flavanoids', 'Nonflav Phenols', 'Proantocyanins', 'Color Intensity', 'Hue', 'OD280/OD315', 'Proline'])
    Y = pd.DataFrame(columns = ['Class','Alcohol', 'Malic Acid', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols', 'Flavanoids', 'Nonflav Phenols', 'Proantocyanins', 'Color Intensity', 'Hue', 'OD280/OD315', 'Proline'])
    X = data[data[column] < median] #put all data less than median in X DF
    Y = data[data[column] >= median] #put all data greater than or equal to median in Y DF

    return X, Y

         
def Gini(data, classes = [1,2,3]):
    data = np.array(data['Class']) #array of all the class values
    acc_gini = 0
    n = len(data)
    if n != 0:
        for label in classes:
            acc_gini+=(sum(data == label)/n)**2 #sum of all instances that belong to each class / the total number of instances
    return 1 - acc_gini


def GiniSplit(data, splitX, splitY):
    n_total = len(data['Class'])
    w_gini = 0.0

    Xn_vi = len(splitX.index)
    Yn_vi = len(splitY.index)

    Xg_vi = Gini(splitX) #gini of the left set
    Yg_vi = Gini(splitY) #gini of the right set

    w_gini = (Xn_vi/n_total)*Xg_vi + (Yn_vi/n_total)*Yg_vi #weighted gini

    return w_gini

    
def stop(data):
    if Gini(data) == 0: #really tried to get the other stop condition
        return True
    # elif GiniSplit(data, X, Y) > Gini(data):
    #     return True
    else:
        return False


def test(row, tree):
    cNode = tree.getRoot()
    pClass = None

    while cNode.left is not None and cNode.right is not None:
        #compare node name and split with row's column name and value
        value = row[1][cNode.name]

        if value < cNode.splitPoint:
            cNode = cNode.left
            pClass = cNode.name

        else:
            cNode = cNode.right
            pClass = cNode.name


    return pClass

    
def accuracy(data):
    correctCount = 0

    for row in data.iterrows():
        if row[1]['Actual'] == row[1]['Predicted']:
            correctCount = correctCount + 1

    accuracy = int((correctCount/len(data.index)) * 100)
    return accuracy


def buildMatrix(data, matrix):
    for row in data.iterrows(): #iterate through data's rows
        act = str((row[1]['Actual'])) #get the actual class as a string
        prd = str((row[1]['Predicted'])) #get the predicted class as a string
        matrix.loc[act][prd] += 1   #go to the [actual][predicted] cell and increment it by 1

    return matrix


def main():
    wineData = importData() #import the data
    wineData.columns = ['Class','Alcohol', 'Malic Acid', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols', 'Flavanoids', 'Nonflav Phenols', 'Proantocyanins', 'Color Intensity', 'Hue', 'OD280/OD315', 'Proline']
    trainDF, valDF = splitData(wineData) #split into training and validation sets

    print("Building tree... please hold...")
    treeChan = DecisionTree() 
    treeChan.grow(trainDF) #building the tree with the training set
    print('\n\n')
    print('"Tree":')
    treeChan.printTree() #really ugly printed tree

    
    print("\n\nTesting validation set...")
    wine = np.empty((0, 2)) #empty array to add the actual and precidted classes of all the validation data

    aClass = None
    pClass = None
    for row in valDF.iterrows(): #iterate through the validation data rows
        aClass = int(row[1]['Class']) #gets the actual class
        pClass = test(row, treeChan) #gets the predicted class with the test function
        newrow = [aClass, pClass] #creates a new row for wine array
        wine = np.vstack([wine, newrow]) #adds the new row to the wine array

    wineClasses = pd.DataFrame(wine, columns = ['Actual', 'Predicted']) #makes a DF out of the wine array because DFs are prettier

    matrixShell = pd.DataFrame(0, index = ['1','2','3'], columns = ['1', '2', '3'])
    matrix = buildMatrix(wineClasses, matrixShell) #builds the confusion matrix

    print('\nConfusion Matrix: Rows = Actual Class, Columns = Predicted Class')
    print(matrix)

    acc = accuracy(wineClasses) #calculates the accuracy 
    print('\nAccuracy: ', acc, '%')
    print('\n\n\n')




if __name__=="__main__": 
    main() 