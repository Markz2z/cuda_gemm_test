#!/bin/bash

nvcc -use_fast_math --ptxas-options="-v" --generate-code=arch=compute_86,code=sm_86 -O3 -I./ -I src/ -lcublas main.cu src/runner.cu -o output

