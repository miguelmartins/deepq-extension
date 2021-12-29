import numpy as np
from enum import IntEnum
from qiskit import QuantumCircuit
from typing import List


class GateEnumerator(IntEnum):
    cx = 0
    rz = 1
    sx = 2
    x = 3


def get_instruction_parameters(operation) -> List[float]:
    name = operation.name
    if GateEnumerator[name] == GateEnumerator.cx:
        return [1.0, -1.0]
    elif GateEnumerator[name] == GateEnumerator.x or GateEnumerator[name] == GateEnumerator.sx:
        return [1.0]
    else:
        return operation.params


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
