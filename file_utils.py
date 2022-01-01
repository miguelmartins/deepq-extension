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
    circuit_files = [f for f in sorted(listdir(filepath)) if isfile(join(filepath, f))]
    final_file = []
    for circuit_file in circuit_files:
        final_file.append(np.load(filepath + circuit_file))
    final_file = np.array(final_file)
    os.makedirs(target_path, exist_ok=True)
    np.save(target_path + target_name, final_file)


def generate_dataset(circuits_path: str, target_path: str) -> np.ndarray:
    """
    Generates a dataset containing families of circuits and the difference in noise between a reference circuit
    C_j_0 and all C_j_i in (FAMILY_{C_j_0}) for 1 <= j <= #FAMILIES
    Args:
        circuits_path: The path to the .npy file containing all circuits
        target_path: The target directory where the .npy files containing noise for each circuit are stored

    Returns:
        np.ndarray of dimensions [#families, #family_size, 2], where the last dimension has both the circuit
        qasm specification and the hammington weight between |0..0> and |ψ>, where |ψ> is the noisy state
        returned by the NISQ device
    """
    circuits = np.load(circuits_path)
    noise_files = [f for f in sorted(listdir(target_path)) if isfile(join(target_path, f))]
    target = []
    for file in noise_files:
        y = np.load(target_path + file)
        target.append(y)
    target = np.array(target)
    return np.stack([circuits, target.reshape(circuits.shape[0], circuits.shape[1])], axis=2)
