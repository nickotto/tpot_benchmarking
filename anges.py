import os
os.chdir('/Users/matsumoton/Git/tpot_benchmarking')
print(os.getcwd())
from tpot.tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from test_utils import extract_labels, get_optimizer, create_dirs


import openml
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset
 # This is done based on the dataset ID.
#dataset = openml.datasets.get_dataset(1164)
#dataset = openml.datasets.get_dataset(1164)
dataset = pd.read_csv("/Users/matsumoton/Documents/anges_cad_1_train.csv",sep=",")
y_train = dataset['target']
X_train = dataset.drop(['target'],axis=1)

test_dataset = pd.read_csv("/Users/matsumoton/Documents/anges_cad_1_test.csv",sep=",")
y_test = test_dataset['target']
X_test = test_dataset.drop(['target'],axis=1)

#X_train, X_test, y_train, y_test = train_test_split(dataset, y, train_size=0.8)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(X_train)
test_img = scaler.transform(X_test)

#test_img = X_test
#train_img = X_train
#from sklearn.decomposition import PCA
#pca = PCA(svd_solver='randomized', iterated_power= 5)
#pca = PCA(n_components = train_img.shape[0])
#pca.fit(train_img)
#train_img = pca.transform(train_img)
#test_img = pca.transform(test_img)

#digits = load_digits()
#X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,train_size=0.75, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#tpot.dump_fitness_tracker('digen25.csv')




# tpot = TPOTClassifier(verbosity=2, max_time_mins=5, population_size=40)
tpot = TPOTClassifier(verbosity=2, population_size=60, generations=20, track_fitnesses=True,
        track_generations=True, resource_logging=True, test_x = test_img, test_y = y_test, scoring="balanced_accuracy") 
#tpot.fit(X_train, y_train)
tpot.fit(train_img, y_train)
tpot.dump_fitness_tracker("/Users/matsumoton/pareto/fitness.csv")
tpot.dump_pareto_fitness_tracker("/Users/matsumoton/pareto/pareto_fitness.csv")
tpot.dump_primitives_mutations("/Users/matsumoton/pareto/mutation_rates.csv")
print(tpot.score(test_img, y_test))