import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    genre = "Horror"
    MODE = 6


    '''
    MODE = 1: (a) Any ten movies of your choice from the MovieLens dataset.
    MODE = 2: (d) Ten movies from the **A** genres you selected in Section 4
    MODE = 3: Movies randomly picked
    MODE = 4: (b) The ten most popular movies
    MODE = 5: (c) The ten best movies
    MODE = 6: Lee's, Devin's, and Mia's lists
    MODE = 7: Movies colored by avg rating
    MODE = 8: Movies colored by num rating
    '''
    genres = ["Movie Id", "Movie Title", "Unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy", "Crime", "Documentary","Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    f = pd.read_table('data/movies.txt', names=genres)
    popmovies, topmovies, avgratings, numratings = load_top() #load popular and top data

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
        lees_list   = [0, 3, 11, 24, 30, 49, 55, 63, 64, 88, 99, 101, 126, 133, 142, 149, 155, 172, 195, 201]
        devins_list = [11, 55, 68, 16, 95, 97, 167, 172, 190, 362, 590, ]
        mias_list   = [0, 7,49,70,81,82,93,94,113,120,150,171,172,180,167,256,274,484,541,819]


    location = 'UVmatrices/'

    for filename in ['basicHW', 'withbiasHW', 'withglobalbiasHW', 'shelf', 'shelfnobias']:
        # print filename
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

        color = [.8, .8, .8]

        if MODE < 6 :
            plt.scatter(V_tilde[0],V_tilde[1], c=color)#"k", alpha = .1)
            for i in range(len(r)):
                plt.scatter(V_tilde[0][r[i]],V_tilde[1][r[i]], label = f["Movie Title"][r[i]])
        elif MODE == 6:
            plt.scatter(V_tilde[0],V_tilde[1], c=color)#"k", alpha = .1)
            plt.scatter(V_tilde[0][lees_list],V_tilde[1][lees_list], label = "Lee's Movies")
            plt.scatter(V_tilde[0][mias_list],V_tilde[1][mias_list], label = "Mia's Movies")
            plt.scatter(V_tilde[0][devins_list],V_tilde[1][devins_list], label = "Devin's Movies")
        elif MODE == 7:
            plt.scatter(V_tilde[0],V_tilde[1], c=avgratings, cmap="plasma")
            plt.colorbar()
        elif MODE == 8:
            highpop = np.where(numratings > 300)[0]
            #plt.scatter(V_tilde[0][highpop], V_tilde[1][highpop], c=numratings[highpop], cmap="plasma")
            plt.scatter(V_tilde[0], V_tilde[1], c=numratings, cmap="plasma")
            plt.colorbar()

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
            title = "2D SVD of movie matrix file: %s with Mia/Devin/Lee movies" %filename
        elif MODE == 7: 
            title = "2D SVD of movie matrix file: %s colored by average rating" %filename
        elif MODE == 8: 
            title = "2D SVD of movie matrix file: %s colored by number of ratings" %filename
        plt.title(title)

        plt.xlabel("SVD Dimension 1 [Arbitrary Units]")
        plt.ylabel("SVD Dimension 2 [Arbitrary Units]")
        plt.legend()
        plt.show()
        plt.savefig("Figures/fig1.png")


        if MODE == 6:
            # fig = plt.subplots(figsize=(5,10))
            genre_locations = []
            column_names = ["Lee", "Devin", "Mia"]
            heatmap = pd.DataFrame(np.zeros((len(genres[2:]), 3)),columns= column_names, index = genres[2:])
            for genre in genres[2:]:
                xgenre = 0.
                ygenre = 0.
                n = 0.
                for i in range(len(f)):
                    if f[genre][i] == 1:
                        xgenre += V_tilde[0][i]
                        ygenre += V_tilde[1][i]
                        n += 1.
                xgenre /= n
                ygenre /= n
                genre_locations.append([xgenre, ygenre])

            for i, lists in enumerate([lees_list, devins_list, mias_list]):
                xavg = np.average(V_tilde[0][lists])
                yavg = np.average(V_tilde[1][lists])
                for j in range(len(genres[2:])):
                    heatmap[column_names[i]][j] = np.dot([xavg, yavg], genre_locations[j])
            sheat = heatmap.sort_values(by = ['Lee'], ascending = False)
            sns.heatmap(sheat, square = True, linewidths=.5)
            plt.title
            plt.show()


def load_top():
    # Load data
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)
    data = np.vstack((Y_train, Y_test))
    movieids    = np.genfromtxt('data/movies.txt', delimiter='\t', usecols=0, dtype='int')
    print(len(movieids))

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
    return popmovies, topmovies, avgratings, numratings

if __name__ == '__main__':
    main()
