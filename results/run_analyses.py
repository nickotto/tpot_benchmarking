import subprocess as sb
from os import sep


if __name__ == "__main__":
    # Standard, SA each generation, SA final population
    results_dir = f".{sep}results{sep}gen_fitnesses{sep}"
    offspring_dir = f".{sep}results{sep}offspring_generation_test{sep}"
    evpop = 20
    evgen = 50
    configname1 = '"Standard"'

    noevpop = 1000
    no_ev_dir = f".{sep}results{sep}gen_fitnesses{sep}"
    ds_info = {'BreastCancer': ('"Breast cancer"', 'Accuracy'),
               'MusicClassification': ('"Music classification"', 'Accuracy'),
               'Concrete': ('Concrete', '"Negative MSE"'),
               'PPB': ('PPB', '"Negative MSE"'),
               'Toxicity': ('Toxicity', '"Negative MSE"'),
               'Bioavailability': ('Bioavailability', '"Negative MSE"')}
    for k in ds_info.keys():
        print(f"\n{k}")
        sb.call(f'python.exe .{sep}results{sep}randomness_test_analyses.py -DE {results_dir} '
                f'-DNE {no_ev_dir} -I {k} -C {configname1} '
                f'--evpop {evpop} --evgen {evgen} --noevpop {noevpop} '
                f'-T {ds_info[k][0]} -F {k} -S {ds_info[k][1]} -R 30',
                shell=True)

        sb.call(f'python.exe .{sep}results{sep}offspring_test_analyses.py -D {offspring_dir} -I {k} '
                f'--evpop {evpop} --evgen {evgen} '
                f'-T {ds_info[k][0]} -F {k} -R 30',
                shell=True)
