import random

# Read the lines from the original file
with open('instance_ids.txt', 'r') as file:
    lines = file.readlines()

# Sample 50 random lines
sampled_lines = random.sample(lines, 50)

# Write the sampled lines to a new file
with open('subset_ids.txt', 'w') as file:
    file.writelines(sampled_lines)
