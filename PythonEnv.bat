#!/usr/bin/env bash
@echo on

cd ..

start python-envs\sample-env\Scripts\activate --"cd Bipedal-Locomotion"

cd "Bipedal-Locomotion"

tensorboard --logdir results
