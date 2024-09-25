# BUERgence

![BUER](logo.png)

BUERgence aims to find the best llama.cpp inference settings to maximize tokens/s.

It's only focused on output (tg) t/s since I don't really care about pp t/s but PRs welcome. 

## Usage

```
python buergence.py --help
```

Right now it has two modes: smart-random and random.

### ```-s smart-random```
Generates random ```-ngl``` and ```-t``` values for the whole search space then dismisses 1/N entries. (N is the value of the ```--dismiss``` parameter). Evaluates everything then, once that's done, search around the best candidate to hopefully find something even better.

### ```-s random``` (not recommended)
Uses (non-repeating) random values and output the best options found until it is terminated or the whole search space has been tested.

Might find good parameters quickly. Or not. Do you feel lucky?

## Results

Random (exhaustive) search:

```
$ time python projects/buergence/buergence.py -m models/Qwen2.5-14B-Instruct-IQ4_XS.gguf -ming 30 -mang 36 -mit 6 -mat 10 -n 64 -r 3 -s random

Best: 8.204321 with -ngl 36 -t 10

real    21m23.730s
user    0m0.000s
sys     0m0.000s
```

Broad smart-random search

```
$ time python projects/buergence/buergence.py -m models/Qwen2.5-14B-Instruct-IQ4_XS.gguf -ming 30 -mang 36 -mit 6 -mat 10 -n 64 -r 3 -s smart-random -sr 2 -di 4

Best: 8.365471 with -ngl 36 -t 12

real    15m1.955s
user    0m0.000s
sys     0m0.015s
```

Narrower smart-random search

```
$ time python projects/buergence/buergence.py -m models/Qwen2.5-14B-Instruct-IQ4_XS.gguf -ming 30 -mang 36 -mit 6 -mat 10 -n 64 -r 3 -s smart-random -sr 1 -di 8 

Best: 8.006471 with -ngl 36 -t 10

real    7m0.285s
user    0m0.000s
sys     0m0.000s
```

Very narrow search

```
time python projects/buergence/buergence.py -m models/Qwen2.5-14B-Instruct-IQ4_XS.gguf -ming 30 -mang 36 -mit 6 -mat 10 -n 64 -r 3 -s smart-random -sr 1 -di 12 

Best: 7.92003 with -ngl 36 -t 9

real    4m17.474s
user    0m0.000s
sys     0m0.000s
```


## TODOs / IDEAS / QUESTIONS
- Read max layers from model file and clamp -max-ngl to that
- Test more parameters
- Allow users to pass params to bench
- Test all the models in a directory
- Experiment with other algorithms like
    - Do a first pass between ```--ngl-min```, ```--ngl-max```, ```--threads-min```, ```--threads-max``` with a big step size. When this is done, find the highest t/s and search around it with finer steps, eventually settling for the best result.
- Optimize for other things for ex:
    - best perf with lowest RAM/VRAM usage
    - best perf with lowest CPU/GPU usage (right now can be emulated with low ```-mat```/```-mang``` values)
    - best perf with lowest I/O load
    - etc
- make a cool TUI with graphs and stuff
- allow tests models with different quants
- use llama-cli directly instead of llama-bench?
- much cleaner code
- separate ```-sr``` into ```-tsr``` and ```-nsr```
- don't re-evaluate best