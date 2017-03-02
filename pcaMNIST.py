#coding: utf-8
import os, sys
import numpy as np
import sklearn.decomposition as decomp
from sklearn.externals import joblib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import utilities as ut
# やること
# n_compo の数値をコマンドライン引数として入力してもらう
# filenameをn_compoをふくめて指定し，同じ名前のファイルが存在する場合，処理をif文でスキップする
# eigenVector directoryを作成する．そこに固有ベクトルの画像を保存する．

# 引数は，固有ベクトルとそれに対応した寄与率（の集まり）
def save_images_EigenVectors(V, E):
    for ii in range(len(V)):
        # plt.figure()を書くと，figure開き過ぎと警告されてしまう
        plt.imshow(V[ii].reshape(28, 28))
        plt.title('Explained_variance_ratio: ' + str(E[ii]*100) + '%')
        plt.savefig('EigenVector' + str( "{0:03d}".format( ii ) ) + '.png')
        plt.clf()
    cumsum_explained = np.cumsum(E)

# クリフープロットを画像として保存する
def save_image_Cliff_plot(filename, E):
    cumsum_explained = np.cumsum(E)
    plt.figure(figsize=(11, 7))
    plt.bar(range(1, len(E)+1), E, align='center')
    plt.title('cliff_plot for MNIST(cumsum_explained: %.2f)' % cumsum_explained[-1] )
    plt.xlabel('EigenValues')
    plt.ylabel('explained ratio')
    plt.savefig(filename)
    plt.figure(figsize=(11, 7))
    plt.bar(range(1, len(E)+1), E, align='center')
    plt.title('cliff_plot for MNIST(cumsum_explained: %.2f)' % cumsum_explained[-1] )
    plt.xlabel('EigenValues')
    plt.ylabel('explained ratio')
    plt.savefig(filename)

def save_image_Cumulative_explained(filename, E):
    cumsum_explained = np.cumsum(E)
    plt.figure(figsize=(11, 7))
    plt.bar(range(1, len(E)+1), cumsum_explained, align='center')
    plt.title('cumsum_explained for MNIST(cumsum_explained: %.2f)' % cumsum_explained[-1] )
    plt.xlabel('EigenValues(cumsum_explained)')
    plt.ylabel('Cumulative explained ratio')
    plt.savefig(filename)

def main():
    args = sys.argv
    # 全データに対して，PCAをかける
    n_compo = int(args[1])
    dirname_model = os.path.join(".", 'model')
    dirname_pic = os.path.join(".", 'eigenVectors')
    dirname_data = os.path.join(".", 'data')
    filename_model = 'model_PCA_D' + str( "{0:03d}".format( n_compo ) ) + '.pkl'
    filename_npz = 'images_pca_D' + str( "{0:03d}".format( n_compo ) ) + '.npz'

    path_cwd = os.getcwd()
    os.chdir(dirname_data)
    imagesTrain, imagesTest, labelsTrain, labelsTest = ut.load_MNIST_npz()
    os.chdir(path_cwd)
    print "Finished Loading train&test dataset."

    # モデルを保存するためのディレクトリがあるかを確認する
    if os.path.isdir(dirname_model):
        pass
    else:
        os.mkdir(dirname_model)
    # 画像を保存するためのディレクトリがあるかを確認する
    if os.path.isdir(dirname_pic):
        pass
    else:
        os.mkdir(dirname_pic)
    # PCAのモデルがシリアライズされて既に存在するかを確認する
    print "Start PCA (Dimension: " + str(n_compo) + ")"
    path_cwd = os.getcwd()
    os.chdir(dirname_model)
    if os.path.isfile(filename_model):
        pca =  joblib.load(filename_model)
    else:
        pca = decomp.PCA(n_components = n_compo)
        pca.fit(imagesTrain)
        joblib.dump(pca, filename_model)
    os.chdir(path_cwd)
    # pcaを掛けたあとのデータ（npzファイル）が存在する場合には，以下のPCAの処理は行わない
    path_cwd = os.getcwd()
    os.chdir(dirname_data)
    if os.path.isfile(filename_npz):
        pass
    else:
        imagesTrain_pca = pca.transform(imagesTrain)
        imagesTest_pca = pca.transform(imagesTest)
        # 次元削減後の各画像データのベクトルをnpzファイルとして出力
        np.savez(filename_npz, train=imagesTrain_pca, test=imagesTest_pca)
    os.chdir(path_cwd)
    # 各固有ベクトルを画像として保存する
    V = pca.components_
    E = pca.explained_variance_ratio_
    path_cwd = os.getcwd()
    os.chdir(dirname_pic)
    save_images_EigenVectors(V, E)
    os.chdir(path_cwd)

    # クリフ−プロット
    filename_pic = 'cliff_plot_D' + str( "{0:03d}".format( n_compo ) ) + '.png'
    dirname = os.path.join(".", 'pic')
    # 画像を保存するためのディレクトリがあるかを確認する
    if os.path.isdir(dirname):
        pass
    else:
        os.mkdir(dirname)
    # 同じ名前の画像ファイルが存在する場合には，以下の処理をスキップする
    print 'Start Cliff-plot.'
    path_cwd = os.getcwd()
    os.chdir(dirname)
    if os.path.isfile(filename_pic):
        pass
    else:
        save_image_Cliff_plot(filename_pic, E)
    os.chdir(path_cwd)

    # 累積寄与率のプロット（寄与率の増加が小さくなったところでうちきり）
    print 'Start plotting Cumulative contribution ratio.'
    filename_pic = 'cumsum_explained_D' + str( "{0:03d}".format( n_compo ) ) + '.png'
    # 同じ名前の画像ファイルが存在する場合には，以下の処理をスキップする
    path_cwd = os.getcwd()
    os.chdir(dirname)
    if os.path.isfile(filename_pic):
        pass
    else:
        save_image_Cumulative_explained(filename_pic, E)
    os.chdir(path_cwd)

if __name__ == '__main__':
    main()
