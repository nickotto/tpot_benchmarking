from sklearn.utils import shuffle
from tpot.tpot import TPOTClassifier, TPOTRegressor
from os import makedirs


def create_dirs(dir_name):
    try:
        makedirs(dir_name)
    except:
        pass


def extract_labels(df, labelname):
    y = df[labelname].copy(deep=True)
    x = df.drop(labelname, axis=1)
    x, y = shuffle(x, y)
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y


def get_optimizer(classification,
                  gens=2,
                  pop_size=10,
                  offspr_size=10,
                  mr=0.9,
                  cr=0.1,
                  scoring=None,
                  cv=5,
                  n_jobs=1,
                  maxtmins=10,
                  verbosity=2,
                  track_fitnesses=False,
                  track_generations=False,
                  resource_logging=False,
                  test_x=None,
                  test_y=None):
    # hp_opt_iterations and hp_opt_mutate_prob are shared among hp tuning each iteration and hp tuning final population
    # hp_improvs_tracker tracks both the hp tuning during each generation and for the final population
    if classification:
        scoring = 'accuracy' if scoring is None else scoring
        pipeline_optimizer = TPOTClassifier(generations=gens, population_size=pop_size, offspring_size=offspr_size,
                                            mutation_rate=mr, crossover_rate=cr, scoring=scoring,
                                            cv=cv, n_jobs=n_jobs, max_eval_time_mins=maxtmins, verbosity=verbosity,
                                            track_fitnesses=track_fitnesses, track_generations=track_generations,
                                            resource_logging=resource_logging, test_x = test_x, test_y = test_y)
    else:
        scoring = 'neg_mean_squared_error' if scoring is None else scoring
        pipeline_optimizer = TPOTRegressor(generations=gens, population_size=pop_size, offspring_size=offspr_size,
                                           mutation_rate=mr, crossover_rate=cr, scoring=scoring,
                                           cv=cv, n_jobs=n_jobs, max_eval_time_mins=maxtmins, verbosity=verbosity,
                                           track_fitnesses=track_fitnesses, track_generations=track_generations,
                                           resource_logging=resource_logging, test_x = test_x, test_y = test_y)
    return pipeline_optimizer
