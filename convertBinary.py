#coding: utf-8
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np

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

    #global rows, cols -> not global
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

def main():
    filename_labels = 'labels_whole.npz'
    filename_images = 'images_whole.npz'
    dirname_data = os.path.join(".", "data")
    # data directory から読み込み，その後書き込みをする．一連の処理が終わると，実行時のディレクトリに戻ってくる．
    if os.path.isdir(dirname_data):
        pass
    else:
        os.mkdir(dirname_data)
    path_cwd = os.getcwd()
    os.chdir(dirname_data)
    if os.path.isfile(filename_labels) and os.path.isfile(filename_images):
        pass
    else:
        imagesTrain, labelsTrain, rows, cols = load_MNIST('training')
        imagesTest, labelsTest, rows, cols = load_MNIST('testing')
        np.savez(filename_labels, train=labelsTrain, test=labelsTest)
        np.savez(filename_images, train=imagesTrain, test=imagesTest)
        print 'Converting finished. : ' + filename_images + ' ' + filename_labels
    os.chdir(path_cwd)

if __name__ == "__main__":
    main()
