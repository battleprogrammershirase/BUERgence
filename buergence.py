import os
import json
import argparse
import subprocess
from tqdm import tqdm
from random import shuffle
from multiprocessing import cpu_count
import time

# TODO: implement smart random search

def random_search(args: dict) -> str:
    search_space = [(x,y) for x in range(args.ngl_min, args.ngl_max+1) for y in range(args.min_threads, args.max_threads+1)]
    shuffle(search_space)
    best = 0.0
    tq = tqdm(search_space, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
    for (ngl, t) in tq:
        tq.set_description(f"Testing -ngl {ngl: <3d} -t {t : <3d}")
        # TODO: get in the right dir etc
        res = subprocess.run(['llama-bench' + '.exe' if os.name == 'nt' else '', '-m', './models/Qwen2.5-0.5B-Instruct-Q6_K_L.gguf', '-ngl', f'{ngl}', '-t', f'{t}', '-o', 'json', '-p', '0', '-n', '64', '-r', '3'], capture_output=True, text=True)
        new_best = json.loads(res.stdout)[0]["avg_ts"]
        new_best = float(new_best)
        if new_best > best:
            best = new_best
            os.system('clear')
            print(f"Best: \x1B[38;2;0;255;0m{best}\x1B[0m with -ngl {ngl} -t {t}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="BUERgence",description="Find the best values for layer offloading and thread count for llama.cpp")
    parser.add_argument('-m', '--model', help='Path of the model to test', required=True)
    parser.add_argument('-d', '--dir', help='Path of the llama.cpp binaries. (Default: .)', default='.')
    parser.add_argument('-mit', '--min-threads', help='Minimum number of CPU threads to test. (Default: 1)', default=1, type=int)
    parser.add_argument('-mat', '--max-threads', help='Maximum number of CPU threads to test. (Default: cpu_count)', default=cpu_count(), type=int)
    parser.add_argument('-ming', '--ngl-min', help='Minimum number of layers to offload to the GPU. (Default: 0)', default=0, type=int)
    # TODO: !!!!!!!!!!!! get default from model file. !!!!!!!!!!!!!!
    parser.add_argument('-mang', '--ngl-max', help='Maximum number of layers to offload to the GPU. (Default: 999)', default=999, type=int)

    args = parser.parse_args()

    random_search(args)