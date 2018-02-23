import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    genre = "Horror"
    MODE = 5

    '''
    MODE = 1: (a) Any ten movies of your choice from the MovieLens dataset.
    MODE = 2: (d) Ten movies from the **A** genres you selected in Section 4
    MODE = 3: Movies randomly picked
    MODE = 4: (b) The ten most popular movies
    MODE = 5: (c) The ten best movies
    '''

    f = pd.read_table('data/movies.txt', names=["Movie Id", "Movie Title", "Unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy", "Crime", "Documentary","Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
    popmovies, topmovies = load_top() #load popular and top data

    if MODE == 1:
        r = [0, 1, 2, 7, 11, 27, 34, 70, 77, 102]
    elif MODE == 2:
        r = []
        for i in range(len(f)):    
            if f[genre][i] == 1:
                r.append(i)
            if len(r) > 9:
                break
    elif MODE == 3:
        r = []
        for _ in range(10):
            r.append(np.random.randint(len(f)))
    elif MODE == 4:
        r = list(popmovies-1)
    elif MODE == 5:
        r = list(topmovies-1)
    elif MODE == 6:
        #Add yo shit here, bitchs:
        lees_list   = []
        mias_list   = [0,7,49,70,81,82,93,94,113,120,150,171,172,180,167,256,274,484,541,819]
        devins_list = [0, 1, 2, 7, 11, 27, 34, 70, 77, 102]


    location = 'UVmatrices/'
    for filename in ['basicHW','withbiasHW', 'withglobalbiasHW', 'shelf']:
        print filename
        # Load from file
        V = np.genfromtxt(location+'V_' + filename + '.txt', dtype='double')
        U = np.genfromtxt(location+'U_' + filename + '.txt', dtype='double')

        # Mean Center all the points along the K axis
        V = V - V.mean(axis = 0)
        U = U - V.mean(axis = 0)

        print("Input shapes: ", V.T.shape, U.T.shape)

        # Compute SVD
        A, s, vh = np.linalg.svd(V.T, full_matrices = False)

        # Take 2 most important axes
        A = A[:,:2]
        print("SVD   shapes: ", A.shape, s.shape, vh.shape)

        # Project All data onto most important axes
        V_tilde = np.dot(A.T, V.T)
        U_tilde = np.dot(A.T, U.T)

        print("Tilde Shapes: ", V_tilde.shape, U_tilde.shape)

        # Make Plots
        plt.scatter(V_tilde[0],V_tilde[1], c='b')

        if MODE < 6 :
            for i in range(len(r)):
                plt.scatter(V_tilde[0][r[i]],V_tilde[1][r[i]], label = f["Movie Title"][r[i]])
        elif MODE == 6:
            plt.scatter(V_tilde[0][lees_list],V_tilde[1][lees_list], label = "Lee's Movies")
            plt.scatter(V_tilde[0][mias_list],V_tilde[1][mias_list], label = "Mia's Movies")
            plt.scatter(V_tilde[0][devins_list],V_tilde[1][devins_list], label = "Devin's Movies")

        if MODE == 1:
            title = "2D SVD of movie matrix file: %s with hand picked movies" %filename
        elif MODE == 2:
            title = "2D SVD of movie matrix file: %s with %s movies" %(filename, genre)
        elif MODE == 3:
            title = "2D SVD of movie matrix file: %s with random movies" %filename
        elif MODE == 4:
            title = "2D SVD of movie matrix file: %s with popular movies" %filename
        elif MODE == 5:
            title = "2D SVD of movie matrix file: %s with top movies" %filename
        elif MODE == 6:
            title = "2D SVD of movie matrix file: %s with top movies" %filename
        plt.title(title)

        plt.xlabel("SVD Dimension 1 [Arbitrary Units]")
        plt.ylabel("SVD Dimension 2 [Arbitrary Units]")
        plt.legend()
        plt.show()



def load_top():
    # Load data
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)
    data = np.vstack((Y_train, Y_test))
    movieids    = np.genfromtxt('data/movies.txt', delimiter='\t', usecols=0, dtype='int')

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
    print("data", data[:,1])
    return popmovies, topmovies

if __name__ == '__main__':
    main()