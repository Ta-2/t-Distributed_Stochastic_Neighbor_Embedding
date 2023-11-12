import numpy as np
from functools  import reduce
import illustrator as ill
import binary_search as bs

#データの取得
X = np.loadtxt('data.csv')
row = 5
col = X.shape[0]

#取得データの確認
print("inputed data:")
print(X)

learning_rate = 0.005
momentum = 0.01
optimaize_loop = 1500
Perplexity = 1.3
threshold = 0.00001
log_perp = np.log2(Perplexity)
print("log_perp: " + str(log_perp))

def d_pow2(vec1, vec2):
    return np.dot((vec1 - vec2), (vec1 - vec2))

def P(X, i, j, sigma):
    d_ij = np.exp( -d_pow2(X[i], X[j])/(2*sigma) )

    def p_ij_sum(l, r):
        if l[0] == r[0]:
            return l
        else:
            i, j = l[0], r[0]
            r_val = np.exp( -d_pow2(X[i], X[j])/(2*sigma) )
            return (l[0], l[1] + r_val)
    
    d_sum =  reduce(p_ij_sum, enumerate(X), (i, 0.0))
    sum = d_sum[1]
    if sum == 0.0:
        return 0.0
    else:
        return d_ij / sum

def Q(X, i, j):
    d_ij = np.exp( -d_pow2(X[i], X[j]) )

    def q_ij_sum(l, r):
        if l[0] == r[0]:
            return l
        else:
            i, j = l[0], r[0]
            r_val = np.exp( -d_pow2(X[i], X[j]) )
            return (l[0], l[1] + r_val)
    
    d_sum =  reduce(q_ij_sum, enumerate(X), (i, 0.0))
    sum = d_sum[1]
    if sum == 0.0:
        return 0.0
    else:
        return d_ij / sum

def Entropy(sigma, X, i):
    num = X.shape[0]
    entropies = [P(X,i,j,sigma) for j in range(num) if i!=j]
    return sum([-p * np.log2(p) for p in entropies if p > 0.0])

def KLdivergence(X, Y, sigmas):
    num = X.shape[0]
    for i in range(num):
        p_ji = np.array([P(X,i,j,sigmas[i]) for j in range(num) if i!=j])
        q_ji = np.array([Q(Y,i,j          ) for j in range(num) if i!=j])

        amount_info = p_ji * np.log2(p_ji / q_ji)

        return sum(amount_info)

sigmas = [
    bs.binary_search(Entropy, (X, i), log_perp, 1.0e-7, [1.0001, 100000.0], 100)
    for i in range(row)
    ]
print("sigmas: " + str(sigmas))

def grad(X, Y, i, sigmas):
    num = X.shape[0]
    p_ji = np.array([P(X, i, j, sigmas[i]) for j in range(num) if i!=j])
    q_ji = np.array([Q(Y, i, j           ) for j in range(num) if i!=j])
    p_ij = np.array([P(X, j, i, sigmas[j]) for j in range(num) if i!=j])
    q_ij = np.array([Q(Y, j, i           ) for j in range(num) if i!=j])
    y_i  = np.array([Y[i] - Y[j]           for j in range(num) if i!=j])
    
    pq = (p_ji - q_ji + p_ij - q_ij).reshape(1, num-1)
    return 2*np.dot(pq, y_i)[0]

def SNE(X, dim):
    num = X.shape[0]
    Y = np.random.rand(row, dim)
    Y_1 = np.zeros([num, dim])
    Y_2 = np.zeros([num, dim])
    
    frames = []
    frames.append(ill.illustrate(Y, margin=0.3))
    cost = KLdivergence(X, Y, sigmas)
    KL = []
    KL.append(cost)
    for i in range(optimaize_loop):
        gradient = np.array([grad(X, Y, i, sigmas) for i in range(num)])
        Y -= learning_rate * gradient
        Y += momentum * (Y_1 - Y_2)
        frames.append(ill.illustrate(Y, margin=0.3))

        new_cost = KLdivergence(X, Y, sigmas)
        KL.append(new_cost)
        if(abs(new_cost - cost) < threshold):
            break

        Y_2 = Y_1
        Y_1 = gradient
        cost = new_cost

    ill.seve_animation(frames, "SNE")
    return Y, KL

compressed_data, KL = SNE(X, 2)

print("KLdivergence: ")
print(KLdivergence(X, compressed_data, sigmas))
ill.line_chart(KL)
ill.show()

ill.illustrate(compressed_data, margin=0.3)
ill.show()