# CZ4052 Assignment 2 - Study Google Pagerank Algorithm and Mapreduce

## Prerequisite

```
pip install -r requirements.txt
```

## Section 1 - Numerical Examples and Experiments

```
# four-webpages example
python pagerank_4_webpages.py
```

```
# a larger network example
# specify the size of the network with -s SIZE
python pagerank_larger.py -s 10000
```

## Section 2 - Exploration of Parameter Tuning

```
# tune the damping factor
# specify the size of the network with -s SIZE
python pagerank_tuning_damping_factor.py -s 10000
```

```
# tune the distribution vector
python pagerank_tuning_distribution_vector.py
```
