## Here is a program to generate training trials for the summer project
## Created by Diheng Zhang (dihengzhang@email.arizona.edu)
## at 6/24/2019

import random
import sys

# Define negative and positive valence here:
ne = 1
po = 1
minimum_act = 2
sparsity = 0.85

# Set the output file
sys.stdout = open('output.txt','wt')

# Keep an eye on the generated trials, make sure there is no replication.
seen = set()

# Generate 180 trials in total
for i in range(0,180):
    sys.stdout.write("_D:	evt_")
    sys.stdout.write(str(i))
    sys.stdout.write("\t")

    #first 60 trial will be negetive
    if i < 60:
        sys.stdout.write(str(ne))
        sys.stdout.write("\t")
        sys.stdout.write("0\t0\t0\t0\t0\t")
    # second 60 trials will be positive
    elif i < 120:
        sys.stdout.write("0\t0\t0\t")
        sys.stdout.write(str(po))
        sys.stdout.write("\t0\t0\t")
    # last 60 trials will be neutral
    else:
        sys.stdout.write("0\t0\t0\t0\t0\t0\t")

    # To generate trials with on replication, spare activations, and above minimum activations.
    is_old = True
    while is_old:
        new_line = ""
        counter = 0
        for j in range(0,25):
            if random.random() > sparsity:
                new_line += '1\t'
                counter += 1
            else:
                new_line += '0\t'
        if (new_line not in seen) and (counter >= minimum_act):
            seen.add(new_line)
            is_old = False

    # print the trials twice for both input and output
    sys.stdout.write(new_line)
    sys.stdout.write(new_line)

    # repeat the BLA inputs for the BLA outputs
    if i < 60:
        sys.stdout.write(str(ne))
        sys.stdout.write("\t")
        sys.stdout.write("0\t0\t0\t0\t0")
    elif i < 120:
        sys.stdout.write("0\t0\t0\t")
        sys.stdout.write(str(po))
        sys.stdout.write("\t0\t0")
    else:
        sys.stdout.write("0\t0\t0\t0\t0\t0")

    sys.stdout.write("\n")
