import os
import re

def rename_files(directory):
    # Pattern to check if a file already has "_kX.npz" at the end
    pattern = re.compile(r"_k\d+\.npz$")
    
    for filename in os.listdir(directory):
        # Process only `.npz` files
        if filename.endswith(".npz") and not pattern.search(filename):
            new_filename = filename.replace(".npz", "_k3.npz")
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

# Specify the directory containing the files
directory = "dat/spline_test/"
rename_files(directory)
