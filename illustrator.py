import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np

#データ描画関数
def illustrate(datas, margin=30):
    #データ数の取得
    data_num = datas.shape[0]
    #プロットデータのリスト
    artists = []

    #各データのクラスを設定
    classes = [chr(i) for i in range(ord('A'), ord('A')+data_num)]

    #各点の色を設定
    r = 0.9*np.linspace(0.1, 0.9, data_num)
    g = 0.9*np.array([0.6]*data_num)
    b = 0.9*np.linspace(0.8, 0.1, data_num)
    color = [(c[0], c[1], c[2]) for c in zip(r, g, b)]

    #グラフの表示範囲を計算
    xmax, xmin = max([d[0] for d in datas]), min([d[0] for d in datas])
    ymax, ymin = max([d[1] for d in datas]), min([d[1] for d in datas])
    
    #点群のプロット
    sct = plt.scatter(x=[d[0] for d in datas], y=[d[1] for d in datas], s=200, c=color)
    artists.append(sct)
    for d, cls in zip(datas, classes):
        tex = plt.text(x=d[0], y=d[1], s=cls, fontsize=18)
        artists.append(tex)

    #その他の表示
    plt.xlim([xmin-margin, xmax+margin])
    plt.ylim([ymin-margin, ymax+margin])
    plt.grid()

    #プロットデータのartistリストを返す
    return artists

#折れ線グラフを描画
def line_chart(datas):
    plt.plot(datas)

#グラフの表示
def show():
    plt.show()

def cla():
    plt.cla()

#アニメーションを保存
def seve_animation(frames, filename):
    ani = ArtistAnimation(plt.gcf(), frames, interval=10)
    ani.save(filename + ".gif", writer='pillow')
    plt.show()