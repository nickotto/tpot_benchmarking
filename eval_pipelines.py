import dill as pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from test_utils import extract_labels, get_optimizer, create_dirs
import os 
import math

import random
import numpy as np

from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import statistics
import dill as pickle
import sys
from tpot import TPOTClassifier
from deap import creator
sys.path.append('/Users/matsumoton/Git/digen')
from digen import Benchmark

benchmark=Benchmark()
plt.rcParams["figure.figsize"] = (30,16)

directoryevs = ["baseline","gi_crossover"]
directoryevs = ["baseline_final","lexicase_final","lexicase_dynamic_final"]

result = {}
for j in [2,4,7,14,23,24,25,27,28,30,32,35,40]:
    print(j)

    dataset=benchmark.load_dataset('digen'+str(j))

    X, Y = extract_labels(dataset, "target")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8,random_state=5)
    ev = []

    tpot = TPOTClassifier(verbosity=2, population_size=1, generations=1, track_fitnesses=True,
        track_generations=True, 
        resource_logging=True,test_x = X_test, test_y = y_test, scoring="balanced_accuracy", cv=10, dynamic_rates = True)
    tpot.fit(X_train, y_train)
    for directoryev in directoryevs:
        print(directoryev)
        upper_ci = []
        lower_ci = []
        test_score = []
        generations = []
        holdout_score = []
        holdout_roc_auc_score = []
        for i in range(40):
            print(i)
            #with open(f"/Users/matsumoton/Git/results_pop40_gen15_{directoryev}/pipelines/digen{j}_run_{i}_evaluated_individuals.pkl", 'rb') as file:
                #unpickler = pickle.Unpickler(file)
                #result = unpickler.load()
            ev_df_name = f"/Users/matsumoton/Git/results_pop40_gen20_{directoryev}/pareto_fitnesses/digen{j}_run_{i}_evolution_pop40_gen20.csv"
            if not exists(ev_df_name):
                continue
            fitness_df = pd.read_csv(ev_df_name, sep=',')
            fitness_df = fitness_df.sort_values(by=['pipeline'])
            prev_pipeline = ''
            for k in range(fitness_df.shape[0]):
                pipeline = fitness_df["pipeline"][k]
                if prev_pipeline == pipeline:
                        generations.append(fitness_df["generation"][k])
                        test_score.append(p)
                        upper_ci.append(p+1.96*s)
                        lower_ci.append(p-1.96*s)
                        holdout_score.append(fitness_df["holdout_score"][k])
                        holdout_roc_auc_score.append(fitness_df["holdout_roc_auc_score"][k])
                        continue
                prev_pipeline = pipeline
                #print(pipeline)
                test = creator.Individual.from_string(pipeline, tpot._pset)
                #print(test)
                pipeline_fitted = tpot._toolbox.compile(expr=test)
                pipeline_fitted.fit(X_train, y_train)
                predictions = pipeline_fitted.predict(X_test)
                p = sum(pipeline_fitted.predict(X_test) == y_test)/len(y_test)
                s = np.sqrt(p*(1-p)/len(y_test))
                generations.append(fitness_df["generation"][k])
                test_score.append(p)
                upper_ci.append(p+1.96*s)
                lower_ci.append(p-1.96*s)
                holdout_score.append(fitness_df["holdout_score"][k])
                holdout_roc_auc_score.append(fitness_df["holdout_roc_auc_score"][k])

        con = pd.DataFrame(np.stack((generations, test_score,upper_ci,lower_ci,holdout_score,holdout_roc_auc_score), axis=1))
        con.columns = ['generations', 'test_score','upper_ci','lower_ci','holdout_score','holdout_roc_auc_score']
        con["type"] = directoryev
        con.to_csv(f"/Users/matsumoton/pareto/{directoryev}_{j}ci.csv", sep=',', index=False)
        
        

                

    
    #median normalized
#    for i in range(0,15):
#        median_gen = statistics.median(frame_df.loc[(frame_df['type']=='baseline')&(frame_df['generation']==i)]['score'])
#        frame_df.loc[frame_df['generation']==i,'score']=frame_df.loc[frame_df['generation']==i]['score'].div(median_gen)

    #for directoryev in directoryevs:
        #seaborn.violinplot(x="generation",y="score",hue="type",data=frame_df, label = "type" if i == 0 else "")
        #plt.show()
        #ax = sns.boxplot(x="generation",y="score",hue="type",data=frame_df)
        #plt.show()
        #ax = sns.swarmplot(x="generation",y="score",hue="type",data=frame_df,color=".25")

    #plt.show()

