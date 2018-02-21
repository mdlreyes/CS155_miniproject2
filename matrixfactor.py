# Matrix factorization for CS 155 Miniproject 2
# Modified from code written by Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
import matplotlib.pyplot as plt

def visualize(Y_train, Y_test):

    data = np.vstack((Y_train, Y_test))

    movieids    = np.genfromtxt('data/movies.txt', delimiter='\t', usecols=0, dtype='int')
    movienames  = np.genfromtxt('data/movies.txt', delimiter='\t', usecols=1, dtype='string')
    moviegenres = np.genfromtxt('data/movies.txt', delimiter='\t', usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20), dtype='int')

    # Make histogram of all ratings in dataset
    plt.figure()
    plt.hist(data[:,2], bins=[0.5,1.5,2.5,3.5,4.5,5.5])
    plt.xlabel('Rating')
    plt.ylabel('N')
    plt.title('All Movies')
    plt.savefig('Figures/allratings.png')

    # Compute average ratings and number of ratings for each movie
    avgratings = np.zeros(len(movieids))
    numratings = np.zeros(len(movieids))

    for i in range(len(data)):
        numratings[data[i,1]-1] += 1.
        avgratings[data[i,1]-1] += data[i,2]

    avgratings = np.divide(avgratings,numratings)

    # Get data for 10 most popular movies
    popmovies = np.argsort(numratings)[-10:] + 1
    popdata = data[np.where(np.in1d(data[:,1],popmovies))]

    # Get data for 10 best movies 
    topmovies = np.argsort(avgratings)[-10:] + 1
    topdata = data[np.where(np.in1d(data[:,1],topmovies))]

    # Make histogram of ratings of top 10 most popular movies
    plt.figure()
    plt.hist(popdata[:,2], bins=[0.5,1.5,2.5,3.5,4.5,5.5])
    plt.xlabel('Rating')
    plt.ylabel('N')
    plt.title('10 Most Popular Movies')
    plt.savefig('Figures/10mostpopratings.png')

    # Make histogram of ratings of 10 best movies
    plt.figure()
    plt.hist(topdata[:,2], bins=[0.5,1.5,2.5,3.5,4.5,5.5])
    plt.xlabel('Rating')
    plt.ylabel('N')
    plt.title('10 Best Movies')
    plt.savefig('Figures/10bestratings.png')

    # Pick genres
    genres = [1,3,5]
    genrelabels = ['Action','Animation','Comedy']

    for j in range(len(genres)):

        # Get all movies in genre
        genremovies = np.where(moviegenres[:,genres[j]])[0] + 1

        genredata = data[np.where(np.in1d(data[:,1],genremovies))]

        # Make histogram of ratings of 10 best movies
        plt.figure()
        plt.hist(genredata[:,2], bins=[0.5,1.5,2.5,3.5,4.5,5.5])
        plt.xlabel('Rating')
        plt.ylabel('N')
        plt.title(genrelabels[j]+' Movies')
        plt.savefig('Figures/'+genrelabels[j]+'ratings.png')

    return Y_train, Y_test
		
def runmodel(bias):
	
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    print("Factorizing with ", M, " users, ", N, " movies.")

    # Parameters
    Ks = 20
    reg = 0.0
    eta = 0.03 # learning rate

    if bias=='simple':
        from matrixfactorbias_utils import train_model, get_err

    elif bias=='global':
        from matrixfactorglobalbias_utils import train_model, get_err

    else:
        from matrixfactor_utils import train_model, get_err

    # Train model and compute training and test error
    if bias=='simple':
        U, V, err, a, b = train_model(M, N, Ks, eta, reg, Y_train)    

        E_train = err
        E_test = get_err(U, V, Y_test, a, b)

    elif bias=='global':
        U, V, err, a, b, mu = train_model(M, N, Ks, eta, reg, Y_train)    

        E_train = err
        E_test = get_err(U, V, Y_test, a, b, mu)

    else:
        U, V, err = train_model(M, N, Ks, eta, reg, Y_train)    

        E_train = err
        E_test = get_err(U, V, Y_test)

    # Save matrices
    if bias=='simple':
        np.savetxt('U_withbiasHW.txt',U)
        np.savetxt('V_withbiasHW.txt',V)
    elif bias=='global':
        np.savetxt('U_withglobalbiasHW.txt',U)
        np.savetxt('V_withglobalbiasHW.txt',V)
    else:
        np.savetxt('U_basicHW.txt',U)
        np.savetxt('V_basicHW.txt',V)

    return

if __name__ == "__main__":

    # Load data
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt')    .astype(int)

    # Make plots
    visualize(Y_train, Y_test)

    # Do the main training
    #runmodel(bias='none')
    #runmodel(bias='simple')
    runmodel(bias='global')