import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from test_utils import extract_labels, get_optimizer, create_dirs
from os import sep
import dill as pickle

import random
import numpy as np

import sys
sys.path.append('./digen')
from digen import Benchmark

#import openml
# Get dataset by ID
#dataset = openml.datasets.get_dataset(61)

# Get dataset by name
#dataset = openml.datasets.get_dataset('Fashion-MNIST')


# Get the data itself as a dataframe (or otherwise)
#X, y, _, _ = dataset.get_data(dataset_format="dataframe")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_num", "-DS", type=int, help="Path and file name of the dataset")
    #parser.add_argument("--labelname", '-L', type=str, help="Name of the column containing the label in the csv")
    parser.add_argument("--scoring", '-S', type=str, default=None, help="scoring metric")
    parser.add_argument("--run_start_id", "-RS", type=int, dest='runstart', default=0, help="run id")
    parser.add_argument("--run_end_id", "-RE", type=int, dest='runend', default=30, help="run id to end at (excluded)")
    parser.add_argument("--pop_random_sampling", "-PR", type=int, default=100, help="Population size for the random sampling")
    parser.add_argument("--pop", "-P", type=int, default=50, help="Population size for TPOT")
    parser.add_argument("--gen", "-G", type=int, default=15, help="Number of generations for the random sampling")
    parser.add_argument("--classification", '-C', dest='classification', action='store_true')
    parser.add_argument("--regression", '-R', dest='classification', action='store_false')
    parser.set_defaults(classification=True)
    args = parser.parse_args()

    #args.dataset = "./datasets/Concrete.csv"
    args.labelname = "target"

    gen_fitnesses_dir = "/common/matsumoton/results/gen_fitnesses" 
    pareto_fitnesses_dir = "/common/matsumoton/results/pareto_fitnesses" 
    offspring_dir = "/common/matsumoton/results/offspring_generation_test"
    resource_logging_dir = "/common/matsumoton/results/resource_logging"
    pipeline_dir = "/common/matsumoton/results/pipelines"
    
    create_dirs(gen_fitnesses_dir)
    create_dirs(pareto_fitnesses_dir)
    create_dirs(offspring_dir)
    create_dirs(resource_logging_dir)
    create_dirs(pipeline_dir)

    pareto_fitnesses_dir = "/common/matsumoton/results_3/pareto_fitnesses" 
    create_dirs(pareto_fitnesses_dir)

    print('digen'+str(args.dataset_num))

    #Downloading a specific dataset
    benchmark=Benchmark()
    args.dataset=benchmark.load_dataset('digen'+str(args.dataset_num))

    #random.seed(5)
    #np.random.seed(5)
    for idx_run in range(args.runstart, args.runend):
        #pipeline_optimizer = get_optimizer(args.classification, gens=args.gen, pop_size=args.pop_random_sampling,
        #                                   offspr_size=args.pop_random_sampling, scoring=args.scoring,
        #                                   track_fitnesses=True,track_generations=True,resource_logging=True)
        #pipeline_optimizer.fit(X_train, Y_train)

        #print("Tpot fit executed. Dumping evolution data into csv")
        #no_ev_dump_file = f"{gen_fitnesses_dir}/{dump_file_name}_no_evolution_pop{args.pop_random_sampling}_gen0.csv"
        #pipeline_optimizer.dump_fitness_tracker(no_ev_dump_file)

        #with open(f"{pipeline_dir}/{dump_file_name}_gen0.pkl", 'wb') as outp:
        #    pickle.dump(pipeline_optimizer, outp, -1)

        # one generation is evaluated outside the number of generations (DEAP based)
        pipeline_optimizer = get_optimizer(args.classification, gens=args.gen - 1, pop_size=args.pop,
                                           offspr_size=args.pop, scoring=args.scoring, track_fitnesses=True,
                                           track_generations=True, resource_logging=True, test_x = X_test, test_y = Y_test,cv=10)

        pipeline_optimizer.fit(X_train, Y_train)

        ev_dump_file = f"{gen_fitnesses_dir}/{dump_file_name}_evolution_pop{args.pop}_gen{args.gen}.csv"
        ev_pareto_dump_file = f"{pareto_fitnesses_dir}/{dump_file_name}_evolution_pop{args.pop}_gen{args.gen}.csv"
        ev_mutation_rate_dump_file = f"{pareto_fitnesses_dir}/{dump_file_name}_evolution_pop{args.pop}_gen{args.gen}_mutrate.csv"

        offspring_dump_file = f"{offspring_dir}/{dump_file_name}_pop{args.pop}_gen{args.pop}"
        resource_logging_dump_file = f"{resource_logging_dir}/{dump_file_name}_pop{args.pop}_gen{args.gen}"
        pipeline_optimizer.dump_fitness_tracker(ev_dump_file)
        pipeline_optimizer.dump_parents_offspring_fitnesses(offspring_dump_file)
        pipeline_optimizer.dump_resource_logging(resource_logging_dump_file)
        pipeline_optimizer.dump_pareto_fitness_tracker(ev_pareto_dump_file)
        pipeline_optimizer.dump_primitives_mutations(ev_mutation_rate_dump_file)

        with open(f"{pipeline_dir}/{dump_file_name}_evaluated_individuals.pkl", 'wb') as outp:
            pickle.dump(pipeline_optimizer.evaluated_individuals_, outp, -1)

