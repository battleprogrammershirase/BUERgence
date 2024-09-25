# BUERgence

![BUER](logo.png)

BUERgence aims to find the best llama.cpp inference settings to maximize tokens/s.

It's only focused on output (tg) t/s since I don't really care about pp t/s but PRs welcome. 

## Usage

Right now it has two modes: random-smart and random.

### -random-smart
Generates random -ngl and -t values for the whole search space then dismisses 1/N entries. (N is the value of the -dismiss parameter). Evaluates everything then once that's done search around the best candidate to hopefully find 

### -random (not recommended)
Uses (non-repeating) random values and output the best options found until it is terminated or the whole search space has been tested.


## TODOs / IDEAS / QUESTIONS
- Read max layers from model file and clamp -max-ngl to that
- Test more parameters
- Allow users to pass params to bench
- Test all the models in a directory
- Experiment with other algorithms like
    - Do a first pass between -ngl-min, -ngl-max, -threads-min, -threads-max with a big step size. When this is done, find the highest t/s and search around it with finer steps, eventually settling for the best result.
- Optimize for other things for ex:
    - best perf with lowest RAM/VRAM usage
    - best perf with lowest CPU/GPU usage (right now can be emulated with low -mat/-mang values)
    - best perf with lowest I/O load
    - etc
- make a cool TUI with graphs and stuff
- allow tests models with different quants
- use llama-cli directly instead of llama-bench?
- much cleaner code
