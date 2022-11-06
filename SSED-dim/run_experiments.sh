#!/usr/bin/bash

#Use this script to run through all configuration files
#in a chosen directory. Script will run all training sessions
#and dump the session outputs as a log in outputs/

cd "$(dirname "$0")"
proj_dir=$(dirname "$0")
experiment_dir="$1"
#Cange experiment dir to run with config files in your chosen directory
for config_file in "${proj_dir}/${experiment_dir}"/*
do
  #Getting conf_name for output logfile
  conf_name="$(basename $config_file)"
  conf_name=$(echo "$conf_name" | cut -f 1 -d '.')
  env/bin/python train.py $config_file > $proj_dir/outputs/$conf_name.log
done
