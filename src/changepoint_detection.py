from numpy import array, flipud, insert, ones, sum, vstack, zeros
from numpy.random import multinomial, uniform
from numpy.random.mtrand import dirichlet
from pylab import axvline, cm, figure, matshow, scatter, show

mean = 200

def hazard(r):
    """
    r is a list of run lengths
    """
    return ones(len(r)) / mean

def categorical(dist):
    return multinomial(1, dist).argmax()

def generate_data(T, K, beta, n=None):
    """
    T is the number of timesteps
    K is the dimensionality
    beta is the concentration parameter
    n is the base measure
    """

    if n == None:
        n = ones(K) / K # uniform base measure

    x = zeros(T) # observations
    r = zeros(T) # run lengths

    C = [] # changepoints

    for t in range(1,T+1):

        # sample run length

        if t == 1:
            r[t-1] = 0
        else:
            if uniform() < hazard([r[t-2] + 1]):
                r[t-1] = 0
            else:
                r[t-1] = r[t-2] + 1

        # sample new parameters if run length is zero

        if r[t-1] == 0:
            C.append(t)
            [phi] = dirichlet(beta * n, 1)

        x[t-1] = categorical(phi) # sample data

    return x, r, C

def inference(x, beta, n=None):
    """
    x is data
    beta is the concentration parameter
    n is the base measure
    """

    T = len(x) # number of timesteps
    K = max(x) + 1 # dimensionality

    if n == None:
        n = ones(K) / K # uniform base measure

    # base case (i.e., initialization) for t=0

    chi = array([beta * n])
    nu = array([beta])
    PR = array([n[x[0]]])

    posteriors = [PR] # posterior distributions over run lengths
    C = [1] # changepoints

    # recursion

    for t in xrange(2, T+1):

        chi = vstack((beta*n, chi))
        chi[1:, x[t-2]] += 1

        nu = insert(nu, 0, beta)
        nu[1:] += 1

        Q = zeros(t)
        Q[0] = n[x[t-1]]
        Q[1:] = chi[1:, x[t-1]] / nu[1:]

        H = hazard(range(1,t))

        S = zeros(t)
        S[0] = sum(H * PR)
        S[1:] = (1 - H) * PR

        PR = Q * S / sum(S)

        posteriors.append(PR / PR.sum())

        if PR.argmax() == 0:
            C.append(t)

    return posteriors, C

def main():

    T=1000 # number of time steps
    K=10 # dimensionality
    beta = 0.01 # concentration parameter

    x, r, true_C = generate_data(T, K, beta)
    posteriors, inferred_C = inference(x, beta)

    print 'True changepoints:', true_C
    print 'Inferred changepoints:', inferred_C

    # plot posterior distributions over run lengths

    matrix = zeros((T,T))

    for i in xrange(T):
        matrix[:i+1, i] = posteriors[i]

    matrix = flipud(matrix)
    matshow(matrix, 0, cmap=cm.bone_r)

    for t in true_C:
        axvline(x=t, color='r')

    # plot data, true changepoints, and inferred changepoints

    figure()

    scatter(range(len(x)), x, alpha=0.3, s=1)

    for t in true_C:
        axvline(x=t, linewidth=4, color='r')

    for t in inferred_C:
        axvline(x=t, linewidth=2, color='k')

    show()

if __name__ == '__main__':
    main()

