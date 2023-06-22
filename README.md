# Evaluation Scores for Elliptic Extended Object Tracking Considering Diverse Object Sizes

Repository containing code to reproduce the experiments presented in
```
Evaluation Scores for Elliptic Extended Object Tracking Considering Diverse Object Sizes
Simon Steuernagel, Kolja Thormann and Marcus Baum
```

To quickly get an overview of all produced figures, take a look at [this folder](figures/presentation).

## Repository Structure

Experiments can be found in [src/paper](src/paper). To simply run all experiments in batch and generate all figures,
you can directly run [this script](src/paper/_run_all.py).

The individual experiments save corresponding files into the [figures](figures) folder. This folder is split into two 
subfolders. One is used to save .svg files for latex embedding ([figures/paper](figures/paper)), the other one saves
png for presentations etc. ([figures/presentation](figures/presentation)).

The corresponding graphics are generated using their respective matplotlib stylesheet, which can be found 
[here](data/stylesheets).
