import os
os.chdir('C:/Users/matsumoton/Box/tpot_benchmarking')
print(os.getcwd())
from tpot.tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from test_utils import extract_labels, get_optimizer, create_dirs


import pandas as pd
 # This is done based on the dataset ID.
#dataset = openml.datasets.get_dataset(1164)
#dataset = openml.datasets.get_dataset(1164)
dataset = pd.read_csv("C:/Users/matsumoton/Box/anges/anges_cad_1_train.csv",sep=",")
y_train = dataset['target']
X_train = dataset.drop(['target'],axis=1)

test_dataset = pd.read_csv("C:/Users/matsumoton/Box/anges/anges_cad_1_test.csv",sep=",")
y_test = test_dataset['target']
X_test = test_dataset.drop(['target'],axis=1)



for run_id in range(3,40):
        # tpot = TPOTClassifier(verbosity=2, max_time_mins=5, population_size=40)
        tpot = TPOTClassifier(verbosity=2, population_size=100, offspring_size=50, generations=50, track_fitnesses=True,
                track_generations=True, resource_logging=True, test_x = X_test, test_y = y_test, scoring="balanced_accuracy",cv=10) 
        #tpot.fit(X_train, y_train)
        tpot.fit(X_train, y_train)
        tpot.dump_fitness_tracker(f"C:/Users/matsumoton/Box/anges/baseline_{run_id}_fitness.csv")
        tpot.dump_pareto_fitness_tracker(f"C:/Users/matsumoton/Box/anges/baseline_{run_id}_pareto_fitness.csv")
        tpot.dump_primitives_mutations(f"C:/Users/matsumoton/Box/anges/baseline_{run_id}_mutation_rates.csv")
        print(tpot.score(X_test, y_test))