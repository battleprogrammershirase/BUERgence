import os
import json
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path
from random import shuffle
from multiprocessing import cpu_count

def smart_random(args):
    llama_bench_path = Path.cwd().joinpath(Path(args.dir+'/llama-bench' + '.exe' if os.name == 'nt' else ''))
    model_path = Path.cwd().joinpath(Path(args.model))
    search_space = [(x,y) for x in range(args.ngl_min, args.ngl_max+1) for y in range(args.min_threads, args.max_threads+1)]
    shuffle(search_space)
    search_space = search_space[0:len(search_space)//args.dismiss]

    best = 0.0
    best_ngl = 0
    best_t = 0
    tq = tqdm(search_space, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
    for (ngl, t) in tq:
        tq.set_description(f"Testing -ngl {ngl: <3d} -t {t : <3d}")
        res = subprocess.run([llama_bench_path, '-m', model_path, '-ngl', f'{ngl}', '-t', f'{t}', '-o', 'json', '-p', '0', '-n', f'{args.n_gen}', '-r', f'{args.repeat}'], capture_output=True, text=True)
        new_best = json.loads(res.stdout)[0]["avg_ts"]
        new_best = float(new_best)
        if new_best > best:
            best = new_best
            best_ngl = ngl
            best_t = t
            os.system('clear')
            print(f"Best: \x1B[38;2;0;255;0m{best}\x1B[0m with -ngl {ngl} -t {t}")

    mig = clamp(best_ngl-args.smart_range, 0, args.ngl_max)
    mag = clamp(best_ngl+args.smart_range, 0, args.ngl_max) ### reading layers from models will fix this

    mit = clamp(best_t-args.smart_range, 0, cpu_count()) ## no chance  mit > cpu_count is better, right?
    mat = clamp(best_t+args.smart_range, 0, cpu_count())

    search_space = [(x,y) for x in range(mig, mag+1) for y in range(mit, mat+1)]

    tq = tqdm(search_space, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
    for (ngl, t) in tq:
        tq.set_description(f"Testing -ngl {ngl: <3d} -t {t : <3d}")
        res = subprocess.run([llama_bench_path, '-m', model_path, '-ngl', f'{ngl}', '-t', f'{t}', '-o', 'json', '-p', '0', '-n', f'{args.n_gen}', '-r', f'{args.repeat}'], capture_output=True, text=True)
        new_best = json.loads(res.stdout)[0]["avg_ts"]
        new_best = float(new_best)
        if new_best > best:
            best = new_best
            best_ngl = ngl
            best_t = t
            os.system('clear')
            print(f"Best: \x1B[38;2;0;255;0m{best}\x1B[0m with -ngl {ngl} -t {t}")


def clamp(n, s, l):
    return max(s, min(n, l))

def random_search(args):
    llama_bench_path = Path.cwd().joinpath(Path(args.dir+'/llama-bench' + '.exe' if os.name == 'nt' else ''))
    model_path = Path.cwd().joinpath(Path(args.model))
    search_space = [(x,y) for x in range(args.ngl_min, args.ngl_max+1) for y in range(args.min_threads, args.max_threads+1)]
    shuffle(search_space)
    best = 0.0
    tq = tqdm(search_space, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
    for (ngl, t) in tq:
        tq.set_description(f"Testing -ngl {ngl: <3d} -t {t : <3d}")
        res = subprocess.run([llama_bench_path, '-m', model_path, '-ngl', f'{ngl}', '-t', f'{t}', '-o', 'json', '-p', '0', '-n', f'{args.n_gen}', '-r', f'{args.repeat}'], capture_output=True, text=True)
        new_best = json.loads(res.stdout)[0]["avg_ts"]
        new_best = float(new_best)
        if new_best > best:
            best = new_best
            os.system('clear')
            print(f"Best: \x1B[38;2;0;255;0m{best}\x1B[0m with -ngl {ngl} -t {t}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="BUERgence",description="Find the best values for layer offloading and thread count for llama.cpp")
    parser.add_argument('-s', '--strategy', help='Strategy to use to search the parameter space. (Default: smart-random)', default="smart-random", choices=['random', 'smart-random'])
    parser.add_argument('-m', '--model', help='Path of the model to test', required=True)
    parser.add_argument('-d', '--dir', help='Path of the llama.cpp binaries. (Default: .)', default='.')
    parser.add_argument('-mit', '--min-threads', help='Minimum number of CPU threads to test. (Default: 1)', default=1, type=int)
    parser.add_argument('-mat', '--max-threads', help='Maximum number of CPU threads to test. (Default: cpu_count)', default=cpu_count(), type=int)
    parser.add_argument('-ming', '--ngl-min', help='Minimum number of layers to offload to the GPU. (Default: 0)', default=0, type=int)
    # TODO: !!!!!!!!!!!! get default from model file. !!!!!!!!!!!!!!
    parser.add_argument('-mang', '--ngl-max', help='Maximum number of layers to offload to the GPU. (Default: 999)', default=999, type=int)
    parser.add_argument('-r', '--repeat', help='How many time to repeat each test. (Default 5, see llama-bench --help)', default=5, type=int)
    parser.add_argument('-n', '--n-gen', help='Number of tokens to gen. (Default 128, see llama-bench --help)', default=128, type=int)
    parser.add_argument('-sr', '--smart-range', help='Range to search around best params found when using smart-random. (Default: 5)', default=5, type=int)
    parser.add_argument('-di', '--dismiss', help='How many entries to dismiss from the search space when using smart-random. For example if the search space is 100 a -di value of 20 will result in a new search space of site 100/5 = 20. (Default: 5)', default=5, type=int)

    args = parser.parse_args()

    match args.strategy:
        case 'smart-random':
            smart_random(args)
        case 'random':
            random_search(args)