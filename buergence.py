from tqdm import tqdm
import subprocess
from random import shuffle
import os
import json

# TODO: parameter handling, implement smart random search

def random_search(params: dict) -> str:
    search_space = [(x+10,y+1) for x in range(params['ngl']) for y in range(params['t']+1)]
    shuffle(search_space)
    best = 0.0
    tq = tqdm(search_space, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
    for (ngl, t) in tq:
        tq.set_description(f"Testing -ngl {ngl: <3d} -t {t : <3d}")
        # TODO: get in the right dir etc
        res = subprocess.run(['llama-bench' + '.exe' if os.name == 'nt' else '', '-m', './models/Qwen2.5-0.5B-Instruct-Q6_K_L.gguf', '-ngl', f'{ngl}', '-t', f'{t}', '-o', 'json', '-p', '0', '-n', '32', '-r', '1'], capture_output=True, text=True)
        new_best = json.loads(res.stdout)[0]["avg_ts"]
        new_best = float(new_best)
        if new_best > best:
            best = new_best
            os.system('clear')
            print(f"Best: \x1B[38;2;0;255;0m{best}\x1B[0m with -ngl {ngl} -t {t}")


if __name__ == '__main__':
    params = {
        'ngl': 42,
         't': 10
    }
    os.system('clear')
    random_search(params)