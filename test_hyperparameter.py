import os
os.chdir('/Users/matsumoton/Git/tpot_benchmarking')
print(os.getcwd())
from tpot.tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from test_utils import extract_labels, get_optimizer, create_dirs

import sys
sys.path.append('/Users/matsumoton/Git/digen')
from digen import Benchmark

benchmark=Benchmark()
dataset=benchmark.load_dataset('digen25')


X, Y = extract_labels(dataset, "target")
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8,random_state=5)

#digits = load_digits()
#X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape



# tpot = TPOTClassifier(verbosity=2, max_time_mins=5, population_size=40)
tpot = TPOTClassifier(verbosity=2, max_time_mins=50, population_size=40, generations=15, track_fitnesses=True,
        track_generations=True,
        resource_logging=True,test_x = X_test, test_y = y_test, scoring="balanced_accuracy", cv=10, dynamic_rates = True)
        #, periodic_checkpoint_folder="/Users/matsumoton/pareto/") 
tpot.fit(X_train, y_train)

tpot.dump_fitness_tracker("/Users/matsumoton/pareto/fitness.csv")
tpot.dump_pareto_fitness_tracker("/Users/matsumoton/pareto/pareto_fitness.csv")
tpot.dump_primitives_mutations("/Users/matsumoton/pareto/mutation_rates.csv")

print(tpot.score(X_test, y_test))
#tpot.dump_fitness_tracker('digen25.csv')

