#!/bin/bash

# output equi.csv, pars.csv
echo Run pd.py
/opt/miniconda3/envs/comp/bin/python /Users/zhugchzo/Desktop/3paper_code/chick/analyze_equation_220/find_equi.py

# Run bifurcation continuation using AUTO and output b.out files for each varied parameter
# (Make sure AUTO runs using Python 2)
echo Run run_cont.auto
auto /Users/zhugchzo/Desktop/3paper_code/chick/analyze_equation_220/run_cont.auto

