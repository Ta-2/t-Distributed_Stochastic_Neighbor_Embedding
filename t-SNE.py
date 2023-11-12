import numpy as np
from functools  import reduce
import illustrator as ill
import binary_search as bs

#データの取得
X = np.loadtxt('data.csv')
row = 5
col = X.shape[0]

#各種パラメータの初期化
#学習率
learning_rate = 0.005
#モメンタム係数
momentum = 0.01
#最適化ループ上限
optimaize_loop = 1500
#各データが揃える情報量の値
Perplexity = 1.3
log_perp = np.log2(Perplexity)
print("log_perp: " + str(log_perp))
#最適化の閾値
threshold = 0.00001

#ベクトルの長さの二乗を返す関数
def d_pow2(vec1, vec2):
    return np.dot((vec1 - vec2), (vec1 - vec2))

#次元削減前の各データから見た確率分布
def P(X, i, j, sigma):
    #指定のデータの確率
    d_ij = np.exp( -d_pow2(X[i], X[j])/(2*sigma) )

    #その他のデータの確率（同一データの確率は排除）
    def p_ij_sum(l, r):
        if l[0] == r[0]:
            return l
        else:
            i, j = l[0], r[0]
            r_val = np.exp( -d_pow2(X[i], X[j])/(2*sigma) )
            return (l[0], l[1] + r_val)
    
    #指定のデータの確率を正規化
    d_sum =  reduce(p_ij_sum, enumerate(X), (i, 0.0))
    sum = d_sum[1]
    if sum == 0.0:
        return 0.0
    else:
        return d_ij / sum

#次元削減後の各データから見た確率分布
def Q(X, i, j):
    #指定のデータの確率
    d_ij = np.exp( -d_pow2(X[i], X[j]) )

    #その他のデータの確率（同一データの確率は排除）
    def q_ij_sum(l, r):
        if l[0] == r[0]:
            return l
        else:
            i, j = l[0], r[0]
            r_val = np.exp( -d_pow2(X[i], X[j]) )
            return (l[0], l[1] + r_val)
    
    #指定のデータの確率を正規化
    d_sum =  reduce(q_ij_sum, enumerate(X), (i, 0.0))
    sum = d_sum[1]
    if sum == 0.0:
        return 0.0
    else:
        return d_ij / sum

#各データから見た確率分布のエントロピー計算
def Entropy(sigma, X, i):
    num = X.shape[0]
    entropies = [P(X,i,j,sigma) for j in range(num) if i!=j]
    return sum([-p * np.log2(p) for p in entropies if p > 0.0])

#KLダイバージェンス
def KLdivergence(X, Y, sigmas):
    #データ数の取得
    num = X.shape[0]
    for i in range(num):
        p_ji = np.array([P(X,i,j,sigmas[i]) for j in range(num) if i!=j])
        q_ji = np.array([Q(Y,i,j          ) for j in range(num) if i!=j])

        amount_info = p_ji * np.log2(p_ji / q_ji)

        return sum(amount_info)

#各データのエントロピーがlog_perpと等しくなるsigma(分散)を二分探索
sigmas = [
    bs.binary_search(Entropy, (X, i), log_perp, 1.0e-7, [1.0001, 100000.0], 100)
    for i in range(row)
    ]
print("variance of each data: " + str(sigmas))

#KLダイバージェンスの勾配関数
def grad(X, Y, i, sigmas):
    #データ数の取得
    num = X.shape[0]
    p_ji = np.array([P(X, i, j, sigmas[i]) for j in range(num) if i!=j])
    q_ji = np.array([Q(Y, i, j           ) for j in range(num) if i!=j])
    p_ij = np.array([P(X, j, i, sigmas[j]) for j in range(num) if i!=j])
    q_ij = np.array([Q(Y, j, i           ) for j in range(num) if i!=j])
    y_i  = np.array([Y[i] - Y[j]           for j in range(num) if i!=j])
    
    pq = (p_ji - q_ji + p_ij - q_ij).reshape(1, num-1)
    return 2*np.dot(pq, y_i)[0]

#KLダイバージェンスが最小になるデータを求める
def SNE(X, dim):
    #データ数の取得
    num = X.shape[0]
    #過去のデータを保持
    Y = np.random.rand(row, dim)
    Y_1 = np.zeros([num, dim])
    Y_2 = np.zeros([num, dim])
    
    #アニメーション用のフレームリスト
    frames = []
    frames.append(ill.illustrate(Y, margin=0.3))
    #コストの変化のリスト
    KL = []
    KL.append(KLdivergence(X, Y, sigmas))

    #モメンタム項付き勾配降下法で最適化
    for i in range(optimaize_loop):
        #勾配を計算
        gradient = np.array([grad(X, Y, i, sigmas) for i in range(num)])
        #勾配と慣性による変化
        Y -= learning_rate * gradient
        Y += momentum * (Y_1 - Y_2)
        #フレームの追加
        frames.append(ill.illustrate(Y, margin=0.3))

        #コストの変化をリストに追加
        new_cost = KLdivergence(X, Y, sigmas)
        KL.append(new_cost)

        #収束条件を見たいしていたらbreak
        if(abs(new_cost - cost) < threshold):
            break

        #次のループへの準備
        Y_2 = Y_1
        Y_1 = gradient
        cost = new_cost

    #アニメーションの保存
    ill.seve_animation(frames, "SNE")
    return Y, KL

#SNE法から次元削減後のデータを取得
compressed_data, KL = SNE(X, 2)

#各データの表示
print("KLdivergence value after optimizing: ")
print(KLdivergence(X, compressed_data, sigmas))
ill.line_chart(KL)
ill.show()

ill.illustrate(compressed_data, margin=0.3)
ill.show()