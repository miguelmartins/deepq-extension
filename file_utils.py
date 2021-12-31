import os
from os import listdir
from os.path import isfile, join
import numpy as np


def combine_circuit_files(filepath: str, target_path: str, target_name: str = 'circuits.npy'):
    """

    Args:
        filepath: The absolute file path of the directory containing multiple npy files
        target_path: The derised absolut target path of the output file
        target_name: (optional, default = 'circuits.npy') File name for the output file

    Returns:
            None
    """
    circuit_files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    final_file = []
    for circuit_file in circuit_files:
        final_file.append(np.load(filepath + circuit_file))
    final_file = np.array(final_file)
    os.makedirs(target_path, exist_ok=True)
    np.save(target_path + target_name, final_file)
