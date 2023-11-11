import numpy as np
from functools  import reduce
import illustrator as ill
import eigen_module as em
import binary_search as bs
import matplotlib.pyplot as plt

#データの取得
X = np.loadtxt('data.csv')
row = 5
col = 5

#取得データの確認
print("inputed data:")
print(X)

Perplexity = 1.3
log_perp = np.log2(Perplexity)
print("log_perp: " + str(log_perp))

sigma = 500.0
compressed_data = np.random.rand(row, 2)

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
    entropies = [P(X,i,j,sigma) for j in range(row) if i!=j]
    #print("each entropies: ")
    #print(entropies)
    return sum([-p * np.log2(p) for p in entropies if p > 0.0])

def KLdivergence(X, Y, sigmas):
    for i in range(row):
        p_ji = np.array([P(X,i,j,sigmas[i]) for j in range(row) if i!=j])
        q_ji = np.array([Q(Y,i,j          ) for j in range(row) if i!=j])

        amount_info = p_ji * np.log2(p_ji / q_ji)
        #print(amount_info)

        return sum(amount_info)

sigma = bs.binary_search(Entropy, (X, 0), log_perp, 1.0e-7, [1.0001, 100000.0], 100)
sigmas = [
    bs.binary_search(Entropy, (X, i), log_perp, 1.0e-7, [1.0001, 100000.0], 100)
    for i in range(row)
    ]
print("sigmas: " + str(sigmas))

entropy = [Entropy(sigmas[i],X,i) for i in range(row)]
#print("whole entropy: ")
#print(entropy)

print("KLdivergence: ")
print(KLdivergence(X, compressed_data, sigmas))

def grad(X, Y, i, sigmas):
    p_ji = np.array([P(X, i, j, sigmas[i]) for j in range(row) if i!=j])
    q_ji = np.array([Q(Y, i, j           ) for j in range(row) if i!=j])
    p_ij = np.array([P(X, j, i, sigmas[j]) for j in range(row) if i!=j])
    q_ij = np.array([Q(Y, j, i           ) for j in range(row) if i!=j])
    y_i  = np.array([Y[i] - Y[j]           for j in range(row) if i!=j])
    
    pq = (p_ji - q_ji + p_ij - q_ij).reshape(1, row-1)
    return 2*np.dot(pq, y_i)[0]

KL = []
for i in range(1000):
    gradient = np.array([grad(X, compressed_data, i, sigmas) for i in range(row)])
    compressed_data -= 0.01 * gradient
    #print("KLdivergence: ")
    KL.append(KLdivergence(X, compressed_data, sigmas))


print("KLdivergence: ")
print(KLdivergence(X, compressed_data, sigmas))


plt.plot(KL)
plt.show()

ill.illustrate(compressed_data, margin=0.3)