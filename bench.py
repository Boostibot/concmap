import matplotlib.pyplot as plt 
import numpy as np
import csv 
import math
import sys
import os 
from typing import List,Dict

class Raw_Bench_File:
    name:str = ""
    threads:int = 0
    trials:int = 0
    records:np.ndarray = np.array([])

    def __init__(self, name:str, threads:int, trials:int):
        self.name = name
        self.threads = threads
        self.trials = trials
        self.records = np.zeros(shape=(threads, trials, 3))

class Median_Bench_File:
    name:str = ""
    threads:int = 0
    trials:int = 0
    records:np.ndarray = np.array([])
    
    def __init__(self, name:str, threads:int, trials:int):
        self.name = name
        self.threads = threads
        self.trials = trials
        self.records = np.zeros(shape=(threads, 3))

class Sum_Bench_File:
    name:str = ""
    threads:int = 0
    trials:int = 0
    iters:int = 0
    okays:int = 0
    duration:int = 0 
    
    def __init__(self, name:str, threads:int, trials:int):
        self.name = name
        self.threads = threads
        self.trials = trials

def load_raw_bench_file(path:str) -> List[Raw_Bench_File]:
    def at_or(arr, i, or_val=0, type=float):
        try:
            return type(arr[i])
        except (ValueError, IndexError):
            return or_val

    bench_files = []
    with open(path,'r') as csvfile: 
        rows = csv.reader(csvfile, delimiter = ',') 

        bench_file = None

        prev_chunk_start = 0
        next_chunk_start = 1
        for i, row in enumerate(rows): 
            # first line is a comment
            if i == 0:
                continue

            # next are either blanks, headers, or body
            if i == next_chunk_start:
                # if is blank then assume next is header
                if len(row) == 0:
                    next_chunk_start += 1
                    continue

                name = at_or(row, 0, "", str)
                threads = at_or(row, 1, type=int)
                trials = at_or(row, 2, type=int)

                prev_chunk_start = next_chunk_start
                next_chunk_start = i + 1 + threads
                bench_file = Raw_Bench_File(name, threads, trials)
                bench_files += [bench_file]
            # else is header
            else:
                assert bench_file != None

                th = i - prev_chunk_start - 1
                for tri in range(bench_file.trials):
                    bench_file.records[th, tri, 0] = at_or(row, 3*tri+0)
                    bench_file.records[th, tri, 1] = at_or(row, 3*tri+1)
                    bench_file.records[th, tri, 2] = at_or(row, 3*tri+2)

    return bench_files        

def median_bench_file(be:Raw_Bench_File) -> Median_Bench_File:
    out = Median_Bench_File(be.name, be.threads, be.trials)
    for th in range(be.threads):
        trials = []
        for tri in range(be.trials):
            trials.append(be.records[th, tri, :])

        trials.sort(key=lambda record: record[1]/record[2])
        median = (trials[len(trials)//2] + trials[(len(trials) + 1)//2]) / 2 
        out.records[th, :] = median

    return out

def sum_bench_file(be:Median_Bench_File) -> Sum_Bench_File:
    out = Sum_Bench_File(be.name, be.threads, be.trials)
    sum = np.sum(be.records, axis=0)
    out.iters = sum[0]
    out.okays = sum[1]
    out.duration = sum[2]

    return out

class Grouped_Bench_Files:
    threads: List[int] = []
    trials: List[int] = []
    iters: List[int] = []
    okays: List[int] = []
    duration: List[int] = []

def group_bench_sum_files(bes: List[Sum_Bench_File]) -> Dict[str, Grouped_Bench_Files]:
    if len(bes) == 0:
        return []
    
    same_names = sorted(bes, key=lambda x: x.name)
    grouped:List[List[Sum_Bench_File]] = []
    name = None
    for be in same_names:
        if name != be.name:
            grouped += [[]]
            name = be.name
        grouped[-1] += [be]

    out:Dict[str, Grouped_Bench_Files] = {}
    for group in grouped:
        new = Grouped_Bench_Files()
        new.threads = np.array([be.threads for be in group], dtype=int)
        new.trials = np.array([be.trials for be in group], dtype=int)
        new.iters = np.array([be.iters for be in group], dtype=int)
        new.okays = np.array([be.okays for be in group], dtype=int)
        new.duration = np.array([be.duration for be in group], dtype=float)*1e-9
        out[group[0].name] = new

    return out

raws: List[Raw_Bench_File] = load_raw_bench_file("bench_data/bench.csv")
medians: List[Median_Bench_File] = [median_bench_file(be) for be in raws]
sums: List[Sum_Bench_File] = [sum_bench_file(be) for be in medians]
grouped = group_bench_sum_files(sums)

print(list(grouped.keys()))

def plot_simple_comp(save=None):
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', pad=5)
    
    linewidth = 1.5
    pointsize = 3
    
    names = ["load", "store", "faa", "and", "cas_all", "cas_success", "lock", "read_lock", "write_lock"]
    # names = ["faa_delay000", "faa_delay100", "faa_delay200", "faa_delay400"]
    for name in names:
        if name in grouped:
            group = grouped[name]
            # ax.plot(group.threads, group.okays/(group.duration/group.threads)*1e-6, linestyle='-', marker='s', label=name, markersize=pointsize, linewidth=linewidth)
            ax.plot(group.threads, group.okays/group.duration*1e-6, linestyle='-', marker='s', label=name, linewidth=linewidth)

    ax.set_yscale('log', base=2)
    plt.xticks(list(grouped.values())[0].threads)
    # plt.yticks(Ys)
    plt.xlabel('threads') 
    plt.ylabel('Mops/s', rotation="horizontal") 
    plt.grid()
    plt.legend() 
    if save != None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.0, dpi=180)
    else:
        plt.show() 

# plot_simple_comp("plot.png")
plot_simple_comp("plotlog.png")
