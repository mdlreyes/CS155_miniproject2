import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import Reader
from surprise import Dataset

# Import data
movies = pd.read_table('data/movies.txt', header=None, names=["Movie Id", "Movie Title", "Unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy", "Crime", "Documentary","Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
np.save("movie", movies)

data = pd.read_table('data/data.txt', header=None, names=["user", "movie", "rating"])
np.save("data", data)

train = pd.read_table('data/train.txt',  header=None, names=["user", "movie", "rating"])
np.save("train", train)

test = pd.read_table('data/test.txt', header=None, names=["user", "movie", "rating"])
np.save("test", test)

# Load data
reader = Reader(rating_scale=(1, 5))
ydata = Dataset.load_from_df(data, reader=reader)
fullset = ydata.build_full_trainset()

def runOffShelf(bias=True):
	# Instantiate SVD
	filterer = SVD(biased=bias)

	# Do cross-validation
	cross_validate(filterer, ydata, measures=['RMSE'], cv=5, verbose=True)

	# Run factorization on full dataset
	filterer.fit(fullset)

	# Get matrices
	u = filterer.pu
	v = filterer.qi
	ubias = filterer.bu
	vbias = filterer.bi

	# Save results
	if bias:
		np.savetxt('U_shelf.txt', u)
		np.savetxt('V_shelf.txt', v)
	else:
		np.savetxt('U_shelfnobias.txt', u)
		np.savetxt('V_shelfnobias.txt', v)

if __name__ == '__main__':
    runOffShelf(bias=True)
    runOffShelf(bias=False)