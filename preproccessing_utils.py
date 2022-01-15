import numpy as np
from enum import IntEnum

from qiskit import QuantumCircuit
from typing import List, Tuple


def mod_2pi(angle: float, atol: float = 0) -> float:
    """
    Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π.
    Source:
       https://qiskit.org/documentation/locale/pt_BR/_modules/qiskit/quantum_info/synthesis/one_qubit_decompose.html"""
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    if abs(wrapped - np.pi) < atol:
        wrapped = -np.pi
    return wrapped


class GateEnumerator(IntEnum):
    cx = 0
    rz = 1
    sx = 2
    x = 3
    u = 4


def get_instruction_parameters(operation) -> List[float]:
    name = operation.name
    if GateEnumerator[name] == GateEnumerator.cx:
        return [1.0, -1.0]
    elif GateEnumerator[name] == GateEnumerator.x or GateEnumerator[name] == GateEnumerator.sx:
        return [1.0]
    else:
        return [mod_2pi(param) for param in operation.params]


def circuit_to_tensor(circuit: QuantumCircuit, max_depth: int = 140) -> np.ndarray:
    """
    Args:
        circuit: A QuantumCircuit
        max_depth: The fixed maximum depth for the quantum circuit

    Returns:
        A numpy array representing a tractable tensor of the circuit
    """
    if circuit.depth() > max_depth:
        raise AttributeError('The circuit has depth {} but maximum depth is {}.'.format(circuit.depth(), max_depth))

    circuit.remove_final_measurements()  # remove barrier and measurements
    instructions = circuit[::-1]  # invert order of list to agree with algebraic order
    circuit_tensor = np.zeros((max_depth, circuit.num_qubits, len(GateEnumerator)))  # initialize circuit tensor
    depth = np.zeros(circuit.num_qubits, dtype=np.int32)
    for operation, register, _ in instructions:
        qubits = [qubit.index for qubit in register]
        if GateEnumerator[operation.name] == GateEnumerator.cx:  # one needs to update the depth of the target qubits
            depth[qubits] = [np.max(depth[qubits])] * len(depth[qubits])

        circuit_tensor[depth[qubits],
                       qubits,
                       GateEnumerator[operation.name].value] = get_instruction_parameters(operation)
        depth[qubits] = depth[qubits] + 1
    return circuit_tensor


def circuit_to_tensor_with_u_gates(circuit: QuantumCircuit, max_depth: int = 140) -> np.ndarray:
    """
    Args:
        circuit: A QuantumCircuit
        max_depth: The fixed maximum depth for the quantum circuit

    Returns:
        A numpy array representing a tractable tensor of the circuit
    """
    if circuit.depth() > max_depth:
        raise AttributeError('The circuit has depth {} but maximum depth is {}.'.format(circuit.depth(), max_depth))
    u_indices = np.arange(GateEnumerator['u'].value, (GateEnumerator['u']).value + 3)
    circuit.remove_final_measurements()  # remove barrier and measurements
    instructions = circuit[::-1]  # invert order of list to agree with algebraic order
    circuit_tensor = np.zeros((max_depth, circuit.num_qubits, len(GateEnumerator) + 2))  # initialize circuit tensor
    depth = np.zeros(circuit.num_qubits, dtype=np.int32)
    for operation, register, _ in instructions:
        qubits = [qubit.index for qubit in register]
        if GateEnumerator[operation.name] == GateEnumerator.cx:  # one needs to update the depth of the target qubits
            depth[qubits] = [np.max(depth[qubits])] * len(depth[qubits])

        if GateEnumerator[operation.name] == GateEnumerator.u:
            circuit_tensor[depth[qubits],
                           qubits,
                           u_indices] = get_instruction_parameters(operation)
        else:
            circuit_tensor[depth[qubits],
                           qubits,
                           GateEnumerator[operation.name].value] = get_instruction_parameters(operation)
        depth[qubits] = depth[qubits] + 1
    return circuit_tensor


def qasm_dataset_to_tensor_dataset(dataset: np.ndarray, circuit_dimensions: Tuple[int]) -> np.ndarray:
    tensor_dataset = np.zeros(dataset.shape[:-1] + circuit_dimensions)
    for family in range(dataset.shape[0]):
        for circuit in range(dataset[family].shape[0]):
            qc = QuantumCircuit.from_qasm_str(dataset[family, circuit, 0])
            tensor_dataset[family, circuit, :, :, :] = circuit_to_tensor_with_u_gates(qc)
    return tensor_dataset, dataset[:, :, 1]


def generate_pair_dataset(dataset: np.ndarray) -> np.ndarray:
    pairs = np.zeros(dataset.shape[:2] + (2,) + dataset.shape[2:])
    for family in range(dataset.shape[0]):
        c0 = dataset[family, 0]
        for c_i in range(dataset[family].shape[0]):
            pairs[family, c_i, 0] = c0
            pairs[family, c_i, 1] = dataset[family, c_i]
    return pairs


def generate_concatenated_dataset(dataset: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray]:
    # TODO: investigate whether it is worth it to mirror the dataset, such that the model learns
    # TODO: the relatioship = (error_i - error_j) = -(error_i - error_j)
    concat_dataset = np.zeros(dataset.shape[:2] + (2 * dataset.shape[2],) + dataset.shape[3:])
    target_diff = np.zeros(target.shape)
    for family in range(dataset.shape[0]):
        c0 = dataset[family, 0]
        target_0 = target[family, 0]
        for c_i in range(dataset[family].shape[0]):
            concat_dataset[family, c_i, dataset.shape[2]:] = c0
            concat_dataset[family, c_i, :dataset.shape[2]] = dataset[family, c_i]
            target_diff[family, c_i] = target_0 - target[family, c_i]
    return concat_dataset, target_diff
