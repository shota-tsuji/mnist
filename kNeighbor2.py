#coding: utf-8
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import operator
import sklearn.decomposition as decomp

def euclideanDistance(array_data1, array_data2):
    #distance = .0
    #array_diff = array_data1 - array_data2
    #array_diff = array_diff * array_diff
    #return math.sqrt(sum(array_diff))
    return np.linalg.norm(array_data1 - array_data2)

def getNeighbors(trainingImages, trainingLabels, testImage, k):
    distances = []
    length = len(testImage)
    # データセットの各点に対して，testImageとの距離計算
    for x in range(len(trainingImages)):
        dist = euclideanDistance(testImage, trainingImages[x])
        distances.append((trainingImages[x], trainingLabels[x], dist))
    distances.sort(key=operator.itemgetter(2))
    neighbors = []
    neighbors_labels = []
    # 距離の近い順に先頭からk個，点のデータとってくる
    for x in range(k):
        neighbors.append(distances[x][0])
        neighbors_labels.append(distances[x][1])
    return neighbors, neighbors_labels

def load_MNIST(dataset="training", digits=np.arange(10), path="."):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training' ")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    #global rows, cols
    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
    images = zeros((N, rows*cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    imAndLab = []
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols])
        labels[i] = lbl[ind[i]]
    return images, labels, rows, cols

def getResponse(neighbors, neighbors_labels):
    classVotes = {}
    for x in range(len(neighbors_labels)):
        response = x
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def outputData(images, labels):
    count_label = []
    for label in np.arange(10):
        #b = (labels == label) * 1
        b = (labels == label)*1
        count_label.append(np.count_nonzero(b))
    f = open('numbersOfLabels.txt', 'w')
    f.write(str(images.shape) + '\n')
    f.write(str(np.arange(10)) + '\n')
    f.write(str(count_label) + '\n')
    f.write(str(sum(count_label)) + '\n')
    f.close()

def plotImage(image, label=None):
    plt.imshow( image.reshape(rows, cols), cmap=cm.gray)
    if not (label is None):
        plt.title('label: ' + str(label))
    plt.show()

def getAccuracy(labelsTest, predictions):
    correct = 0
    for x in range(len(labelsTest)):
        if labelsTest[x] is predictions[x]:
            correct += 1
    return ( correct / float(len(labelsTest)) ) *100.0

def stop():
    import sys
    sys.exit(0)

#images, labels = load_mnist('training', digits=[8])
imagesTrain, labelsTrain, rows, cols = load_MNIST('training')
print "loaded training dataset."
imagesTest, labelsTest, rows, cols = load_MNIST('testing')
print "loaded testing dataset."

#outputdata(images1, labels1)
#for x in range(len(images2)):
#for x in range(10):
#    neighbors, neighbors_labels = getneighbors(trainingimages=images1, #traininglabels=labels1, testimage=images2[x], k=5)
#    plotimage(images2[x], label=labels2[x])
#    for (xx, xx_label) in zip(neighbors, neighbors_labels):
#        plotimage(xx, label=xx_label)
#num = input('please input a number')
#print labels[num]
#plt.imshow(images[num, 0:rows, 0:cols])
#plt.imshow(images[num], cm.gray)
#plt.show()
predictions = []
k = 5
# 各Testデータについて見ていく（ラベルを推測する）
for ii in range(len(labelsTest)):
    print 'test ' + str(ii)
    neighbors, neighbors_labels = getNeighbors(imagesTrain, labelsTrain, imagesTest[ii], k=k)
    prediction = getResponse(neighbors=neighbors, neighbors_labels=neighbors_labels)
    predictions.append(prediction)
accuracy = getAccuracy(labelsTest, predictions)
print 'Accuracy: ' + str(accuracy) + '%'
