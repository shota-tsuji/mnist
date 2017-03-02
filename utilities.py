#coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os, sys

# npzファイルになっている元画像データとラベルデータを返す関数
def load_MNIST_npz():
    filename_images = 'images_whole.npz'
    filename_labels = 'labels_whole.npz'
    images_whole = np.load(filename_images)
    labels_whole = np.load(filename_labels)
    #(keys_images1, keys_images2) = images_whole.files
    return  images_whole['train'], images_whole['test'], labels_whole['train'], labels_whole['test']

# npzファイルになっているPCA後の画像データとラベルデータを返す関数
def load_MNIST_pca(filename):
    images_pca = np.load(filename)
    return images_pca['train'], images_pca['test']

# 画像データとラベルデータのサイズなどを書き込む関数
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


# MNISTのイメージをひとつ表示する関数
def plotImage(image, label=None, rows=28, cols=28, cmap=cm.gray):
    plt.imshow( image.reshape(rows, cols), cmap=cmap)
    if not (label is None):
        plt.title('label: ' + str(label))
    plt.show()

# 好きなタイミングでプログラムを終了できる関数
def stop():
    sys.exit(0)

#if __name__ == '__main__':
#    n_compo = 10
#    dirname = os.path.join(".", 'data')
#    filename_npz = 'images_pca_D' + str( "{0:03d}".format( n_compo ) ) + '.npz'
#    path_cwd = os.getcwd()
#    os.chdir(dirname)
#    load_MNIST_pca(filename_npz)
