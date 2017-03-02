#coding: utf-8
import os
import sys
import numpy as np
import sklearn.decomposition as decomp
import operator

import utilities as ut
# やることリスト
# 第一引 : kNNのk
# 第二引数 : pcaのn_compo
# data directoryを見る．n_compo用のデータがあれば読み込む．

# numpyのndarrayを仮定して，ユークリッド距離を計算する関数
def euclideanDistance(ndarray1, ndarray2):
    return np.linalg.norm(ndarray1 - ndarray2)

def getNeighbors(images_Train, labels_Train, image_Test, num_k):
    distances = []
    # テストデータひとつ（ベクトル）に対して，すべてのトレインデータとの距離を計算する
    for x in range( len(images_Train) ):
        dist = euclideanDistance(image_Test, images_Train[x])
        distances.append((images_Train[x], labels_Train[x], dist))
    distances.sort(key=operator.itemgetter(2))
    neighbors = np.empty((0, 784), np.uint8)
    neighbors_labels = np.zeros(num_k, np.uint8)
    # 距離の近い順に先頭からk個，点のデータをとってくる
    for x in range(num_k):
        neighbors = np.append(neighbors, np.array([ distances[x][0] ]))
        neighbors_labels[x] = distances[x][1]
    return neighbors, neighbors_labels

# エラーの原因1(for x in range(len(neighbors_labels))にしていた)
def getResponse(neighbors, neighbors_labels):
    neighbors_labels = neighbors_labels.tolist()
    classVotes = {}
    for x in neighbors_labels:
        response = x
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 多数決とって，最多のラベルを返す
    return sortedVotes[0][0]

# 縦ベクトルでも正しく差がとれる
def getAccuracy(labels_Test, labels_Predictions):
    #print labels_Test
    #print labels_Predictions
    diff = labels_Predictions - labels_Test
    #print 'diff'
    #print diff
    # 誤って推測されたラベルのインデックスを得る
    # 非ゼロ要素のインデックスを得る(要素数１のタプルは','がうしろに付くので，[0]で指定する)
    indexFail = np.where(diff)[0].tolist()
    #print  np.where(diff)[0]
    # 誤って推測されたラベルの個数をカウント
    numFail = len(indexFail)
    # 正答率を計算
    #print len(labels_Predictions)
    correct = float(len(labels_Predictions)) - numFail
    return (correct / len(labels_Predictions)) * 100.0

if __name__ == '__main__':
    dirname_data = os.path.join(".", 'data')
    path_cwd = os.getcwd()
    os.chdir(dirname_data)
    imagesTrain_pca, imagesTest_pca, labelsTrain, labelsTest = ut.load_MNIST_npz()
    os.chdir(path_cwd)

    args = sys.argv
    num_k = int(args[1])
    n_compo = int(args[2])
    ls_print = range(1, len(labelsTest) +1, 1000)
    if n_compo == 0:
        print 'normal knn.'
    else:
        print 'This is KNN (k=' + str(num_k) + ', n_compo=' + str(n_compo) + ').'
        filename_npz = 'images_pca_D' + str( "{0:03d}".format( n_compo ) ) + '.npz'
        path_cwd = os.getcwd()
        os.chdir(dirname_data)
        if os.path.isfile(filename_npz):
            imagesTrain_pca, imagesTest_pca = ut.load_MNIST_pca(filename_npz)
        else:
            raise OSError('There is not such file.: ./data directory')
        os.chdir(path_cwd)

    predictions = []
    print len(labelsTest)
    for ii in range(len(labelsTest)):
        if (ii + 1 ) in ls_print:
            print 'Test: ' + str(ii + 1) + ' / ' + str(len(labelsTest))
        neighbors, neighbors_labels = getNeighbors(imagesTrain_pca, labelsTrain, imagesTest_pca[ii], num_k=num_k)
        #print neighbors_labels
        #ut.plotImage(neighbors[0])
        #ut.plotImage(neighbors[1])
        #ut.plotImage(neighbors[2])
        prediction = getResponse(neighbors=neighbors, neighbors_labels=neighbors_labels)
        predictions.append(prediction)

    # 各テストデータについて見ていく（ラベルを推測する）
    predictions = np.asarray(predictions)
    filename_npz = "predictions_k" + str(num_k) + "_pcaD" + str(n_compo) + ".npz"
    path_cwd = os.getcwd()
    os.chdir(dirname_data)
    if os.path.isfile(filename_npz):
        pass
    else:
        np.savez(filename_npz, pred=predictions)
    os.chdir(path_cwd)

    # エラーの原因1(labelsTestは縦ベクトルでした)
    accuracy = getAccuracy(labelsTest.T, predictions)
    print str(accuracy) + '%'
