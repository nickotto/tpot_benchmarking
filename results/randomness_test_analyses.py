import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from os import makedirs
from scipy.stats import wilcoxon
from statannot import add_stat_annotation


def get_file_name(directory_name, inputname, i, evpop, evgen):
    df_name = f"{directory_name}{inputname}_R{i}_evolution_pop{evpop}_gen{evgen}.csv"
    return df_name


def extract_bests(ev_dfs, nev_dfs):
    # extract bests of each generation of evolutions
    max_gen = max(ev_dfs[0]['generation'])
    bests_ev = [[] for _ in range(max_gen + 1)]
    for df in ev_dfs:
        for gen in range(max_gen + 1):
            gen_scores = df.loc[df['generation'] == gen]['score']
            bests_ev[gen].append(max(gen_scores))
    bests_ev = np.array(bests_ev)
    # extract bests of each random sampling
    bests_nev = []
    for df in nev_dfs:
        bests_nev.append(max(df['score']))
    bests_nev = np.array(bests_nev)
    return bests_ev, bests_nev


def compute_medians_bests(bests_ev, bests_nev):
    # median for evolution
    medians_ev = np.median(bests_ev, axis=1)
    # median for no evolution dfs
    median_nev = np.median(bests_nev)
    return medians_ev, median_nev


def plot_boxplots(last_gen_bests, random_bests, score, configname, title, save_directory, filename):
    df_ev_last_gen = pd.DataFrame({'score': last_gen_bests})
    df_ev_last_gen['type'] = configname
    df_nev = pd.DataFrame({'score': random_bests})
    df_nev['type'] = 'Random sampling'
    df = pd.concat((df_ev_last_gen, df_nev))

    _, pvalue = wilcoxon(last_gen_bests, random_bests,
                         zero_method='wilcox', correction=False,
                         alternative='two-sided', mode='auto')
    print(f"Wilcoxon Two-sided pvalue: {pvalue}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.despine(offset=10, trim=False)
    sns.boxplot(x='type', y='score', data=df, ax=ax, showmeans=False, dodge=False)
    add_stat_annotation(ax, data=df, x='type', y='score', perform_stat_test=False,
                        box_pairs=[(configname, 'Random sampling')],
                        pvalues=[pvalue],
                        pvalue_thresholds=[[1e-3, '**'], [1e-2, '*'],
                                           [0.05, '*'], [1, 'ns']],
                        test=None, text_format='star', loc='outside',
                        line_height=0.02, linewidth=2.5, fontsize=24, verbose=0)
    ax.set_xlabel('')
    ax.set_ylabel(f"{score}", fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=14)
    plt.suptitle(f"{title}", fontsize=30)
    plt.tight_layout()
    plt.savefig(f"{save_directory}/images/pdf/{filename}_distribution.pdf")
    plt.savefig(f"{save_directory}/images/png/{filename}_distribution.png", bbox_inches='tight')


def wrap_compute_to_plot(best_medians, FE_per_gen):
    def compute_to_plot(medians, FE):
        to_plot = []
        xs = []
        x = 0
        for m in medians:
            to_plot.append(m)
            xs.append(x)
            to_plot.append(m)
            x += FE
            xs.append(x)
        return to_plot, xs

    to_plot, xs = compute_to_plot(best_medians, FE_per_gen)
    return to_plot, xs


def plot_convergence(ev_best_medians, nev_best_median, FE_per_gen, score, configname, title, save_directory, filename):
    to_plot_nev = [nev_best_median, nev_best_median]
    to_plot_ev, xs = wrap_compute_to_plot(ev_best_medians, FE_per_gen)

    plt.figure(figsize=(8, 8))
    plt.plot([xs[0], xs[-1]], to_plot_nev, color='orange', lw=2.5, label='Random sampling')
    plt.plot(xs, to_plot_ev, color='blue', lw=2.5, label=configname)
    plt.xlabel("Pipeline evaluation", fontsize=20)
    plt.ylabel(f"{score}", fontsize=20)
    plt.title(f"{title}", fontsize=30)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, xs[-1])
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{save_directory}/images/pdf/{filename}_converegence.pdf")
    plt.savefig(f"{save_directory}/images/png/{filename}_convergence.png")


if __name__ == "__main__":
    # the no evolution number of generations is always zero
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsdirectoryev", "-DE", dest="directoryev", type=str)
    parser.add_argument("--resultsdirectorynev", "-DNE", dest="directorynev", type=str)
    parser.add_argument("--inputname", '-I', type=str)
    parser.add_argument("--evpop", type=int)
    parser.add_argument("--evgen", type=int)
    parser.add_argument("--noevpop", type=int)
    parser.add_argument("--title", '-T', type=str)
    parser.add_argument("--filename", '-F', type=str)
    parser.add_argument("--score", '-S', type=str, default='Accuracy')
    parser.add_argument("--configname", '-C', type=str, help="Label for plots")
    parser.add_argument("--numberofruns", '-R', type=int, default=30)
    args = parser.parse_args()

    ev_dfs = []
    nev_dfs = []
    for i in range(args.numberofruns):
        ev_df_name = get_file_name(args.directoryev, args.inputname, i, args.evpop, args.evgen)
        ev_dfs.append(pd.read_csv(ev_df_name, sep=','))
        nev_df_name = f"{args.directorynev}{args.inputname}_R{i}_no_evolution_pop{args.noevpop}_gen0.csv"
        nev_dfs.append(pd.read_csv(nev_df_name, sep=','))

    try:
        makedirs(f"{args.directoryev}images/pdf/")
        makedirs(f"{args.directoryev}images/png/")
    except:
        pass

    ev_bests, nev_bests = extract_bests(ev_dfs, nev_dfs)
    ev_best_medians, nev_best_median = compute_medians_bests(ev_bests, nev_bests)

    last_gen_bests = [np.max(c) for c in ev_bests.T]
    random_bests = [x for x in nev_bests]
    plot_boxplots(last_gen_bests, random_bests, args.score, args.configname, args.title, args.directoryev, args.filename)

    FE_per_gen = args.evpop

    plot_convergence(ev_best_medians, nev_best_median, FE_per_gen, args.score, args.configname, args.title, args.directoryev,
                     args.filename)
