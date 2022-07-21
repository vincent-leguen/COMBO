#!/bin/bash
sbatch  --time=200:00:00  --mem 150000 --gres=gpu:1 --partition=gpu   run.sh