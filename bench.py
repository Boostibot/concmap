import matplotlib.pyplot as plt 
import numpy as np
import csv 
import math
import sys
import os 
import inspect
from typing import List,Dict,Any,Tuple

def group(arr: list, key:Any, astype=list, sort=False)-> List[list] | Dict[Any, list]: 
    grouped = []

    if sort:
        arr = sorted(arr, key=key)
    sig = inspect.signature(key)
    prev = None
    first = True
    if len(sig.parameters) == 1:
        if astype == list:
            for item in arr:
                new_prop = key(item)
                if first or prev != new_prop:
                    grouped.append([])
                    prev = new_prop
                    first = False
                grouped[-1].append(item)
        elif astype == dict:
            grouped = {}
            for item in arr:
                new_prop = key(item)
                if first or prev != new_prop:
                    grouped[new_prop] = []
                    prev = new_prop
                    first = False
                grouped[new_prop].append(item)
    elif len(sig.parameters) == 1:
        for item in arr:
            if first or bool(key(prev, item)):
                grouped.append([])
                prev = item
                first = False
            grouped[-1].append(item)
    else:
        raise TypeError("bad function args")
    
    return grouped

def average(arr:Any, by:Any=None):
    if by == None:
        return sum(arr)/len(arr)
    else:
        sum_map = sum(map(by, arr))
        return sum_map/len(arr)

def median(arr:Any, key:Any=None):
    l = len(arr)
    if l == 0:
        return None
    
    s = sorted(arr, key=key)
    s0 = s[(l - 1)//2]
    s1 = s[l//2]
    if isinstance(s0, str):
        return s0
    else:
        return (s0 + s1) / 2 

class Bench_Result:
    bench_name:str = ""
    thread_name:str = ""
    thread_index:int = 0
    threads:int = 0
    trials:int = 0
    summed:int = 1
    records:List[np.ndarray] = []

def load_bench_result(path:str) -> Tuple[List[Bench_Result], List[Any]]:
    errors=[]
    bench_files = []
    def at_or(arr, i, or_val, astype=None):
        if astype == None:
            astype = type(or_val)
        try:
            return astype(arr[i])
        except (ValueError, IndexError) as e:
            nonlocal errors
            errors.append((arr, i, or_val, astype, e))
            return or_val

    with open(path,'r') as csvfile: 
        rows = csv.reader(csvfile, delimiter = ',', skipinitialspace=True) 
        had_first = False
        for i, row in enumerate(rows): 
            # Allow blanks
            if len(row) == 0:
                continue

            # first line is a comment
            if had_first == False and len(row) <= 1:
                had_first = True
                continue

            # regular line 
            else:
                had_first = True

                be = Bench_Result() 
                be.bench_name = at_or(row, 0, "ERROR")
                be.thread_name = at_or(row, 1, "ERROR")
                be.thread_index = at_or(row, 2, 0)
                be.threads = at_or(row, 3, 0)
                be.trials = at_or(row, 4, 0)
                be.records = []
                for tri in range(be.trials):
                    record = np.array([
                        at_or(row, 5 + 3*tri+0, 0),
                        at_or(row, 5 + 3*tri+1, 0),
                        at_or(row, 5 + 3*tri+2, 0),
                    ])
                    be.records.append(record)

                bench_files.append(be)

    return (bench_files, errors)        

def median_bench_result(be:Bench_Result) -> Bench_Result:
    out = Bench_Result()
    out.bench_name = be.bench_name
    out.thread_name = be.thread_name
    out.thread_index = be.thread_index
    out.threads = be.threads
    out.trials = be.trials
    out.records = [median(be.records, key=lambda record: record[1]/record[2])]
    return out

def sum_bench(bes:List[Bench_Result]) -> Bench_Result:
    out = Bench_Result()
    if len(bes) == 0:
        return out
    
    out.bench_name = median([be.bench_name for be in bes])
    out.thread_name = median([be.thread_name for be in bes])
    out.thread_index = median([be.thread_index for be in bes])
    out.threads = median([be.threads for be in bes])
    out.trials = median([be.trials for be in bes])
    out.records = [sum([sum(be.records) for be in bes])]
    out.summed = len(bes)
    return out

class Grouped_Bench_Files:
    threads: List[int] = []
    trials: List[int] = []
    summed: List[int] = []
    iters: List[int] = []
    okays: List[int] = []
    duration: List[int] = []

def group_sum_bench_files(bes: List[Bench_Result]) -> Dict[str, Dict[str, Grouped_Bench_Files]]:
    grouped_by_benchmark = {}
    
    # if is not medianed already, median
    med = [median_bench_result(be) for be in bes]

    # group by benchmark name and iterate each group
    by_bench = group(med, lambda be: be.bench_name, astype=dict, sort=True)
    for bench_name,benchmark in by_bench.items():

        # group each bencmark by thread name
        grouped_by_thread_name = {}
        by_name = group(benchmark, lambda be: be.thread_name, astype=dict, sort=True)
        for thread_name,thread in by_name.items():
            
            # finally group each thread by number of threads and sum within each category
            by_threads = group(thread, lambda be: be.threads, astype=list, sort=True)
            summed:List[Bench_Result] = [sum_bench(by_thread) for by_thread in by_threads]

            # transpose summed list for easier work
            g = Grouped_Bench_Files()
            g.threads = np.array([s.threads for s in summed])
            g.trials = np.array([s.trials for s in summed])
            g.summed = np.array([s.summed for s in summed])
            g.iters = np.array([s.records[0][0] for s in summed])
            g.okays = np.array([s.records[0][1] for s in summed])
            g.duration = np.array([s.records[0][2] for s in summed])*1e-9

            grouped_by_thread_name[thread_name] = g
        grouped_by_benchmark[bench_name] = grouped_by_thread_name

    return grouped_by_benchmark



def deref(map, arr_or_key):
    if isinstance(arr_or_key, list):
        curr = map
        for a in arr_or_key:
            curr = curr[a]
        return curr
    else:
        return map[arr_or_key]
    
def simple_plot(grouped, names, labels=None, save=None, scale="normal", mode="throughput", title=None):
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', pad=5)
    
    linewidth = 1.5
    pointsize = 3
    
    ticks = []
    for i,name in enumerate(names):
        try:
            group = deref(grouped, name)
            if isinstance(group, dict) == False:
                group = {"":group}

            for k,v in group.items():
                ticks = v.threads
                label = labels[i] if labels != None else str(name) + k
                to_plot = []
                if mode == "throughput":
                    to_plot = v.okays/(v.duration/v.summed)
                elif mode == "efficiency":
                    to_plot = v.okays/(v.duration)

                ax.plot(v.threads, to_plot*1e-6, linestyle='-', marker='s', label=label, markersize=pointsize, linewidth=linewidth)
        except IndexError as e:
            print(f"Error: {e=}")
            None

    if scale=="log":
        ax.set_yscale('log', base=2)
    plt.xticks(ticks)
    # plt.yticks(Ys)
    plt.xlabel('threads') 
    plt.ylabel('Mops/s', rotation="horizontal") 
    if title != None:
        plt.title(title)
    plt.grid()
    plt.legend() 
    if save != None:
        plt.savefig(save, bbox_inches='tight', dpi=180)
    else:
        plt.show() 


records_atomic, errors_atomic = load_bench_result("bench_data/bench_atomic.csv")
records_delay, errors_delay = load_bench_result("bench_data/bench_delay.csv")
records_pressure, errors_pressure = load_bench_result("bench_data/bench_pressure.csv")

redords = records_atomic + records_delay + records_pressure
grouped = group_sum_bench_files(redords)

group_names = {bench_name:list(benchmark.keys()) for bench_name,benchmark in grouped.items()}
print(group_names)

save = True
simple_plot(
    grouped, 
    names= ["load", "store", "exchange", "faa", "and", "bts", "cas_all", "cas_success", "lock", "read_lock", "write_lock"],
    save="bench_data/ops_all_with_load_store.png" if save else None
)
simple_plot(
    grouped, 
    names= ["faa", "exchange", "and", "bts", "cas_all", "cas_success", "lock", "read_lock", "write_lock"], 
    save="bench_data/ops_all.png" if save else None
)
simple_plot(grouped, ["cas_delay0", "cas_delay1", "cas_delay2", "cas_delay4"])

simple_plot(
    grouped,  
    names = [["store_loadp", "op"], ["exchange_loadp", "op"], ["faa_loadp", "op"], ["and_loadp", "op"], ["bts_loadp", "op"], ["cas_loadp", "op"], ["write_lock_loadp", "op"]],
    labels= ["1 store N-1 load", "1 exch N-1 load", "1 and N-1 load", "1 bts N-1 load", "1 cas N-1 load", "1 write lock N-1 read lock"],
    title="Throughput under load pressure (plotting the atomic op throughput)",
    save="bench_data/plot_load_pressure_with_store.png" if save else None
)

simple_plot(
    grouped,  
    names = [["store_loadp", "load"], ["exchange_loadp", "load"], ["faa_loadp", "load"], ["and_loadp", "load"], ["bts_loadp", "load"], ["cas_loadp", "load"], ["write_lock_loadp", "load"]],
    labels= ["N-1 load 1 store", "N-1 load 1 exch", "N-1 load 1 and", "N-1 load 1 bts", "N-1 load 1 cas", "N-1 read lock 1 write lock"],
    title="Throughput under load pressure (plotting the load throughput)",
    save="bench_data/plot_load_pressure_loads_with_store.png" if save else None
)

simple_plot(
    grouped,  
    names = [["exchange_loadp", "op"], ["faa_loadp", "op"], ["and_loadp", "op"], ["bts_loadp", "op"], ["cas_loadp", "op"], ["write_lock_loadp", "op"]],
    labels= ["1 exch N-1 load", "1 and N-1 load", "1 bts N-1 load", "1 cas N-1 load", "1 write lock N-1 read lock"],
    title="Throughput under load pressure (plotting the atomic op throughput)",
    save="bench_data/plot_load_pressure.png" if save else None
)

simple_plot(
    grouped,  
    names = [["exchange_loadp", "load"], ["faa_loadp", "load"], ["and_loadp", "load"], ["bts_loadp", "load"], ["cas_loadp", "load"], ["write_lock_loadp", "load"]],
    labels= ["N-1 load 1 exch", "N-1 load 1 and", "N-1 load 1 bts", "N-1 load 1 cas", "N-1 read lock 1 write lock"],
    title="Throughput under load pressure (plotting the load throughput)",
    save="bench_data/plot_load_pressure_loads.png" if save else None
)
