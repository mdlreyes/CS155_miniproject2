# Matrix factorization for CS 155 Miniproject 2
# Modified from code written by Fabian Boemer, Sid Murching, Suraj Nair

# Most complex matrix factorization (include bias terms a, b, mu)

import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta, ai, bj, mu):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """

    grad1 = reg*Ui
    grad2 = 2.*Vj*(Yij-mu - np.dot(Ui,Vj)-ai-bj)

    return eta*(grad1-grad2)

def grad_V(Vj, Yij, Ui, reg, eta, ai, bj, mu):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    grad1 = reg*Vj
    grad2 = 2.*Ui*(Yij-mu - np.dot(Ui,Vj)-ai-bj)
    
    return eta*(grad1-grad2)

def grad_a(Ui, Yij, Vj, reg, eta, ai, bj, mu):

    grad1 = reg*ai
    grad2 = 2.*(Yij-mu - np.dot(Ui,Vj)-ai-bj)

    return eta*(grad1-grad2)

def grad_b(Ui, Yij, Vj, reg, eta, ai, bj, mu):
    grad1 = reg*bj
    grad2 = 2.*(Yij-mu - np.dot(Ui,Vj)-ai-bj)

    return eta*(grad1-grad2)

def get_err(U, V, Y, a, b, mu, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """

    UF = np.sqrt(np.sum(np.power(U,2.)))
    VF = np.sqrt(np.sum(np.power(V,2.)))
    aF = np.sqrt(np.sum(np.power(V,2.)))
    bF = np.sqrt(np.sum(np.power(V,2.)))
    regterm = reg/2. * (UF**2. + VF**2. + aF**2. + bF**2.)

    errterm = 0.
    for nTerm in range(len(Y)):
        i = Y[nTerm][0] - 1
        j = Y[nTerm][1] - 1

        errterm += 0.5*(Y[nTerm][2] - mu - np.dot(U[i,:],V.T[:,j]) - a[i] - b[j])**2.

    return regterm + errterm


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """

    # Initialize U and V randomly
    U = np.random.uniform(-0.5,0.5,(M,K))
    V = np.random.uniform(-0.5,0.5,(N,K))

    # Initialize a and b randomly
    a = np.random.uniform(-0.5,0.5,(M))
    b = np.random.uniform(-0.5,0.5,(N))

    # Compute global bias (avg over all observed Y)
    mu = np.mean(Y[:,2])

    nEpoch = 0
    while (nEpoch < max_epochs):

        # Shuffle training data indices
        index = np.random.permutation(len(Y))
        Y_shuffle = Y[index]

        # Calculate old error
        if nEpoch == 0:
            errOld = get_err(U,V,Y,a,b, mu)
        else:
            errOld = errNew

        # Do SGD
        for nTerm in range(len(Y)):

            # Get indices
            i = Y_shuffle[nTerm][0] - 1
            j = Y_shuffle[nTerm][1] - 1

            # Update row of U and column of V
            U[i,:] -= grad_U(U[i,:], Y_shuffle[nTerm][2], V.T[:,j], reg, eta, a[i], b[j], mu)
            V.T[:,j] -= grad_V(V.T[:,j], Y_shuffle[nTerm][2], U[i,:], reg, eta, a[i], b[j], mu)

            # Update bias terms
            a[i] -= grad_a(U[i,:], Y_shuffle[nTerm][2], V.T[:,j], reg, eta, a[i], b[j], mu)
            b[j] -= grad_b(U[i,:], Y_shuffle[nTerm][2], V.T[:,j], reg, eta, a[i], b[j], mu)

        # Calculate new error
        errNew = get_err(U,V,Y,a,b,mu)

        if nEpoch == 0:

            # Calculate decrease in MSE
            MSEdec = errOld - errNew
            print eps*MSEdec
            nEpoch += 1

        else:
            # Compare decrease in MSE to first epoch
            if (errOld - errNew) > (eps*MSEdec):
                print errOld - errNew
                nEpoch += 1
            else:
                break

    return U, V, a, b, mu