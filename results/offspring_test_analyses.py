import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import makedirs
from matplotlib.lines import Line2D


def get_files_name(directory_name, inputname, i, evpop, evgen):
    cx_name = f"{directory_name}{inputname}_R{i}_pop{evpop}_gen{evgen}_crossover.csv"
    mut_name = f"{directory_name}{inputname}_R{i}_pop{evpop}_gen{evgen}_mutation.csv"
    return cx_name, mut_name


def determine_better(row):
    if row.offspring <= row.parent1 and row.offspring <= row.parent2:
        return 'worse'
    if row.offspring <= row.parent1 or row.offspring <= row.parent2:
        return 'better than one'
    return 'better than two'


def count_improvement(cx_dfs, mut_dfs):
    cx_counters = [{'worse': 0, 'better than one': 0, 'better than two': 0} for _ in range(len(cx_dfs))]
    for cx_c, cx_df in zip(cx_counters, cx_dfs):
        for row in cx_df.itertuples(index=False):
            offspring_better = determine_better(row)
            cx_c[offspring_better] += 1

    mut_counters = [{'worse': 0, 'better': 0} for _ in range(len(mut_dfs))]
    for mut_c, mut_df in zip(mut_counters, mut_dfs):
        for row in mut_df.itertuples(index=False):
            if row.offspring <= row.parent:
                mut_c['worse'] += 1
            else:
                mut_c['better'] += 1

    return cx_counters, mut_counters


def compute_p_each_generation(cx_dfs, mut_dfs, max_gen):
    cx_ps = [[] for _ in range(1, max_gen)]
    mut_ps = [[] for _ in range(1, max_gen)]
    for g in range(1, max_gen):
        for cx_df in cx_dfs:
            tmp_df = cx_df.loc[cx_df['gen'] == g]
            if tmp_df.shape[0] != 0:
                n_increment = 0
                for row in tmp_df.itertuples(index=False):
                    offspring_better = determine_better(row)
                    if offspring_better != 'worse':
                        n_increment += 1
                p = n_increment / tmp_df.shape[0]
                cx_ps[g - 1].append(p * 100)
        for mut_df in mut_dfs:
            tmp_df = mut_df.loc[mut_df['gen'] == g]
            if tmp_df.shape[0] != 0:
                n_increment = 0
                for row in tmp_df.itertuples(index=False):
                    n_increment += 1 if row.offspring >= row.parent else 0
                p = n_increment / tmp_df.shape[0]
                mut_ps[g - 1].append(p * 100)
    return cx_ps, mut_ps


def compute_aggregate_metrics(counters):
    keys = [k for k in counters[0]]
    aggregate = {k: [] for k in keys}
    for cx_c in counters:
        for k in keys:
            aggregate[k].append(cx_c[k])
    aggregate_metrics = {k: {'mean': np.mean(aggregate[k]),
                             'std': np.std(aggregate[k]),
                             'median': np.median(aggregate[k])}
                         for k in aggregate}
    return aggregate_metrics


def aggregate_metrics(cx_counters, mut_counters):
    cx_metrics = compute_aggregate_metrics(cx_counters)
    mut_metrics = compute_aggregate_metrics(mut_counters)
    return cx_metrics, mut_metrics


def plot_improvements(c_cx, c_mut, title, save_directory, filename):
    plt.figure(figsize=(8, 8))
    plt.bar([1, 2, 3],
            [c_cx['worse']['mean'], c_cx['better than one']['mean'], c_cx['better than two']['mean']],
            alpha=0.9)
    # plot medians
    for x, v in zip([1, 2, 3],
                    [c_cx['worse']['median'], c_cx['better than one']['median'], c_cx['better than two']['median']]):
        plt.plot([x - 0.35, x + 0.35], [v, v], lw=2.5, color='orange', alpha=0.9, zorder=11)
    # plot stds
    for x, m, std in zip([1, 2, 3],
                         [c_cx['worse']['mean'], c_cx['better than one']['mean'], c_cx['better than two']['mean']],
                         [c_cx['worse']['std'], c_cx['better than one']['std'], c_cx['better than two']['std']]):
        plt.plot([x - 0.1, x + 0.1], [m - std, m - std], lw=1.5, ls='-', color='black', alpha=0.8, zorder=10)
        plt.plot([x - 0.1, x + 0.1], [m + std, m + std], lw=1.5, ls='-', color='black', alpha=0.8, zorder=10)
        plt.plot([x, x], [m - std, m + std], ls='-', lw=1.5, color='black', alpha=0.8, zorder=10)
    plt.xticks([1, 2, 3],
               ["No improvement", "Improved one parent", "Improved both parents"],
               rotation=30, fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"{title} - Crossover", fontsize=26)
    plt.legend(handles=[Line2D([0], [0], lw=2.5, label='Mean'),
                        Line2D([0], [0], color='orange', lw=2.5, label='Median'),
                        Line2D([0], [0], color='black', lw=1.5, label='Standard Deviation')],
               fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_directory}/images/pdf/{filename}_cx.pdf")
    plt.savefig(f"{save_directory}/images/png/{filename}_cx.png")

    plt.figure(figsize=(8, 8))
    plt.bar([1, 2],
            [c_mut['worse']['mean'], c_mut['better']['mean']],
            alpha=0.9)
    # plot medians
    for x, v in zip([1, 2], [c_mut['worse']['median'], c_mut['better']['median']]):
        plt.plot([x - 0.35, x + 0.35], [v, v], lw=2.5, color='orange', alpha=0.9, zorder=11)
    # plot stds
    for x, m, std in zip([1, 2],
                         [c_mut['worse']['mean'], c_mut['better']['mean']],
                         [c_mut['worse']['std'], c_mut['better']['std']]):
        plt.plot([x - 0.1, x + 0.1], [m - std, m - std], lw=1.5, ls='-', color='black', alpha=0.8, zorder=10)
        plt.plot([x - 0.1, x + 0.1], [m + std, m + std], lw=1.5, ls='-', color='black', alpha=0.8, zorder=10)
        plt.plot([x, x], [m - std, m + std], ls='-', lw=1.5, color='black', alpha=0.8, zorder=10)
    plt.xticks([1, 2], ["No improvement", "Improved"],
               rotation=30, fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"{title} - Mutation", fontsize=26)
    plt.legend(handles=[Line2D([0], [0], lw=2.5, label='Mean'),
                        Line2D([0], [0], color='orange', lw=2.5, label='Median'),
                        Line2D([0], [0], color='black', lw=1.5, label='Standard Deviation')],
               fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{save_directory}/images/pdf/{filename}_mut.pdf")
    plt.savefig(f"{save_directory}/images/png/{filename}_mut.png")


def percentage_metrics(ps):
    means = []
    stds = []
    for gen_p in ps:
        means.append(np.mean(gen_p))
        stds.append(np.std(gen_p))
    return means, stds


def plot_percentage_improvement_vs_generation(cx_p_improvement, mut_p_improvement, max_gen,
                                              title, save_directory, filename):
    cx_mean, cx_std = percentage_metrics(cx_p_improvement)
    mut_mean, mut_std = percentage_metrics(mut_p_improvement)
    plt.figure(figsize=(8, 8))
    plt.plot([i for i in range(1, max_gen)], cx_mean, color='dimgrey', lw=2.5, label="Crossover")
    plt.plot([i for i in range(1, max_gen)], mut_mean, color='darkgreen', lw=2.5, label="Mutation")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=18)
    plt.xlabel("Generation", fontsize=20)
    plt.ylabel("Improvement percentage", fontsize=20)
    plt.xlim(1, max_gen - 1)
    plt.ylim(0)
    plt.title(f"{title}", fontsize=30)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{save_directory}/images/pdf/{filename}_percentage.pdf")
    plt.savefig(f"{save_directory}/images/png/{filename}_percentage.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsdirectory", "-D", dest="directory", type=str)
    parser.add_argument("--inputname", '-I', type=str)
    parser.add_argument("--evpop", type=int)
    parser.add_argument("--evgen", type=int)
    parser.add_argument("--title", '-T', type=str)
    parser.add_argument("--filename", '-F', type=str)
    parser.add_argument("--numberofruns", '-R', type=int, default=30)
    args = parser.parse_args()

    cx_dfs = []
    mut_dfs = []
    for i in range(args.numberofruns):
        cx_df_name, mut_df_name = get_files_name(args.directory, args.inputname, i, args.evpop, args.evgen)
        cx_dfs.append(pd.read_csv(cx_df_name, sep=','))
        mut_dfs.append(pd.read_csv(mut_df_name, sep=','))
    try:
        makedirs(f"{args.directory}images/pdf/")
        makedirs(f"{args.directory}images/png/")
    except:
        pass

    cx_counters, mut_counters = count_improvement(cx_dfs, mut_dfs)
    cx_metrics, mut_metrics = aggregate_metrics(cx_counters, mut_counters)
    print(cx_metrics)
    print(mut_metrics)
    plot_improvements(cx_metrics, mut_metrics, args.title, args.directory, args.filename)

    cx_p_improvement, mut_p_improvement = compute_p_each_generation(cx_dfs, mut_dfs, args.evgen)
    plot_percentage_improvement_vs_generation(cx_p_improvement, mut_p_improvement, args.evgen,
                                              args.title, args.directory, args.filename)
