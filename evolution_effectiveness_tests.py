import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from test_utils import extract_labels, get_optimizer, create_dirs
from os import sep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-DS", type=str, help="Path and file name of the dataset")
    parser.add_argument("--labelname", '-L', type=str, help="Name of the column containing the label in the csv")
    parser.add_argument("--scoring", '-S', type=str, default=None, help="scoring metric")
    parser.add_argument("--run_start_id", "-RS", type=int, dest='runstart', default=0, help="run id")
    parser.add_argument("--run_end_id", "-RE", type=int, dest='runend', default=30, help="run id to end at (excluded)")
    parser.add_argument("--pop_random_sampling", type=int, default=1000, help="Population size for the random sampling")
    parser.add_argument("--pop", type=int, default=20, help="Population size for TPOT")
    parser.add_argument("--gen", type=int, default=50, help="Number of generations for the random sampling")
    parser.add_argument("--classification", '-C', dest='classification', action='store_true')
    parser.add_argument("--regression", '-R', dest='classification', action='store_false')
    parser.set_defaults(classification=True)
    args = parser.parse_args()

    args.dataset = "./datasets/Concrete.csv"
    args.labelname = "label"
    gen_fitnesses_dir = "./results/gen_fitnesses"
    offspring_dir = "./results/offspring_generation_test/"
    resource_logging_dir = "./results/resource_logging/"
    create_dirs(gen_fitnesses_dir)
    create_dirs(offspring_dir)
    create_dirs(resource_logging_dir)

    for idx_run in range(args.runstart, args.runend):
        print(idx_run)
        df = pd.read_csv(args.dataset, sep=',')
        dump_file_name = f"{args.dataset.split(sep)[-1].split('.')[0]}_R{idx_run}"
        #X, Y = extract_labels(df, args.labelname)
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
        from sklearn.datasets import load_digits
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.75, test_size=0.25)
        X_train.shape, X_test.shape, y_train.shape, y_test.shape

        pipeline_optimizer = get_optimizer(args.classification, gens=0, pop_size=args.pop_random_sampling,
                                           offspr_size=args.pop_random_sampling, scoring=args.scoring,
                                           track_fitnesses=True,track_generations=True,resource_logging=True)
        pipeline_optimizer.fit(X_train, y_train)
        no_ev_dump_file = f"{gen_fitnesses_dir}/{dump_file_name}_no_evolution_pop{args.pop_random_sampling}_gen0.csv"
        pipeline_optimizer.dump_fitness_tracker(no_ev_dump_file)

        # one generation is evaluated outside the number of generations (DEAP based)
        pipeline_optimizer = get_optimizer(args.classification, gens=args.gen - 1, pop_size=args.pop,
                                           offspr_size=args.pop, scoring=args.scoring, track_fitnesses=True,
                                           track_generations=True, resource_logging=True)
        pipeline_optimizer.fit(X_train, y_train)

        ev_dump_file = f"{gen_fitnesses_dir}/{dump_file_name}_evolution_pop{args.pop}_gen{args.gen}.csv"
        offspring_dump_file = f"{offspring_dir}/{dump_file_name}_pop{args.pop}_gen{args.pop}"
        resource_logging_dump_file = f"{resource_logging_dir}/{dump_file_name}_pop{args.pop}_gen{args.gen}"
        pipeline_optimizer.dump_fitness_tracker(ev_dump_file)
        pipeline_optimizer.dump_parents_offspring_fitnesses(offspring_dump_file)
        pipeline_optimizer.dump_resource_logging(resource_logging_dump_file)
