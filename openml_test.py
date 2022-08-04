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
dataset = openml.datasets.get_dataset(1164)

# Print a summary
print(
    f"This is dataset '{dataset.name}', the target feature is "
    f"'{dataset.default_target_attribute}'"
)
print(f"URL: {dataset.url}")
print(dataset.description[:500])

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=5)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(X_train)
test_img = scaler.transform(X_test)

from sklearn.decomposition import PCA
#pca = PCA(svd_solver='randomized', iterated_power= 5)
pca = PCA(n_components = train_img.shape[0])
pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

#digits = load_digits()
#X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape



# tpot = TPOTClassifier(verbosity=2, max_time_mins=5, population_size=40)
tpot = TPOTClassifier(verbosity=2, population_size=40, generations=40, track_fitnesses=True,
        track_generations=True, resource_logging=True, test_x = test_img, test_y = y_test) 
#tpot.fit(X_train, y_train)
tpot.fit(train_img, y_train)
tpot.dump_fitness_tracker("/Users/matsumoton/pareto/fitness.csv")
tpot.dump_pareto_fitness_tracker("/Users/matsumoton/pareto/pareto_fitness.csv")
tpot.dump_primitives_mutations("/Users/matsumoton/pareto/mutation_rates.csv")
print(tpot.score(test_img, y_test))
#tpot.dump_fitness_tracker('digen25.csv')