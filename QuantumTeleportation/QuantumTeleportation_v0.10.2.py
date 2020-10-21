import netsquid as ns
import netsquid.components.instructions as instr
from netsquid.nodes import Node, Network, DirectConnection
from netsquid.components import QuantumChannel, QuantumProgram, ClassicalChannel
from netsquid.protocols import NodeProtocol
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel


def create_processor(depolar_rate, dephase_rate):
    """Factory to create a quantum processor for each end node.

    Has three memory positions and the physical instructions necessary
    for teleportation.
    """
    measure_noise_model = DephaseNoiseModel(dephase_rate=dephase_rate,
                                            time_independent=True)
    physical_instructions = [
        PhysicalInstruction(instr.INSTR_INIT, duration=3, parallel=True),
        PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True, topology=[0, 1]),
        PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(instr.INSTR_S, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False, topology=[0],
                            q_noise_model=measure_noise_model, apply_q_noise_after=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False, topology=[1])
    ]
    # memory_noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
    processor = QuantumProcessor("quantum_processor",
                                 num_positions=3,
                                 # memory_noise_models=[memory_noise_model] * 2,
                                 phys_instructions=physical_instructions)
    return processor


class InitStateProgram(QuantumProgram):
    """
    Program to create a qubit and transform it to the |1> state.
    """

    default_num_qubits = 1

    def program(self):
        q1, = self.get_qubit_indices(1)
        self.apply(instr.INSTR_INIT, q1)
        self.apply(instr.INSTR_X, q1)
        yield self.run()


class GenerateEntanglement(QuantumProgram):
    """
    Program to create two qubits and entangle them.
    """

    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_INIT, [q1, q2])
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_CNOT, [q1, q2])
        yield self.run()


class BellMeasurementProgram(QuantumProgram):
    """
    Program to perform a Bell measurement on two qubits.
    Measurement results are stored in output keys "M1" and "M2"
    """

    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_CNOT, [q1, q2])
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_MEASURE, q1, output_key="M1")
        self.apply(instr.INSTR_MEASURE, q2, output_key="M2")
        yield self.run()


class TeleportEncoderProtocol(NodeProtocol):
    """
    Protocol for the encoder node performing teleportation.
    """

    def __init__(self, node, port_names=("portQA_1", "portCA_1")):
        super().__init__(node)
        self.node = node
        self.q_port = port_names[0]
        self.c_port = port_names[1]

    def run(self):
        # Await qubit preparation
        self.node.qmemory.execute_program(InitStateProgram())
        yield self.await_program(self.node.qmemory)

        self.node.qmemory.execute_program(GenerateEntanglement(), qubit_mapping=[1, 2])
        yield self.await_program(self.node.qmemory)

        # Send entangled half
        self.node.ports[self.q_port].tx_output(self.node.qmemory.pop(2))

        # Do Bell measurements
        measure_program = BellMeasurementProgram()
        self.node.qmemory.execute_program(measure_program, qubit_mapping=[0, 1])
        yield self.await_program(self.node.qmemory)

        # Send measurement results classically
        m1, = measure_program.output["M1"]
        m2, = measure_program.output["M2"]
        self.node.ports[self.c_port].tx_output((m1, m2))


class TeleportDecoderProtocol(NodeProtocol):
    """
    Protocol for the decoder node performing teleportation.
    """

    def __init__(self, node, port_names=("portQB_1", "portCB_1")):
        super().__init__(node)
        self.node = node
        self.q_port = port_names[0]
        self.c_port = port_names[1]

    def run(self):
        # Await entanglement half
        yield self.await_port_input(self.node.ports[self.q_port])

        # Await classical
        yield self.await_port_input(self.node.ports[self.c_port])
        meas_res, = self.node.ports[self.c_port].rx_input().items

        # Perform corrections
        if meas_res[0] == 1:
            self.node.qmemory.execute_instruction(instr.INSTR_Z)
            yield self.await_program(self.node.qmemory)
        if meas_res[1] == 1:
            self.node.qmemory.execute_instruction(instr.INSTR_X)
            yield self.await_program(self.node.qmemory)

        # Measure the qubit to ensure it was in expected state (|1> in this case)
        t = self.node.qmemory.execute_instruction(instr.INSTR_MEASURE, output_key="M")
        yield self.await_program(self.node.qmemory)

        # Print the measurement result
        print('Bob measured teleported qubit in state:', t[0]['M'])


def generate_network():
    """
    Generate the network. There are two nodes in the network connected with a quantum
    and classical channel, both unidirectional from A -> B.
    """

    q_chan = QuantumChannel(name='AqB')
    c_chan = ClassicalChannel(name='AcB')

    network = Network('Teleport Network')

    alice = Node('alice', qmemory=create_processor(1e7, 0.2))
    bob = Node('bob', qmemory=create_processor(1e7, 0.2))

    network.add_nodes([alice, bob])

    _, p_ba = network.add_connection(alice,
                                     bob,
                                     label='q_chan',
                                     connection=DirectConnection(name="q_conn[A|B]",
                                                                 channel_AtoB=q_chan),
                                     port_name_node1='portQA_1',
                                     port_name_node2='portQB_1')

    # Map the qubit input port from the above channel to the memory index 0 on Bob's
    # side
    bob.ports[p_ba].forward_input(bob.qmemory.ports['qin0'])

    network.add_connection(alice,
                           bob,
                           label='c_chan',
                           connection=DirectConnection(name="c_conn[A|B]",
                                                       channel_AtoB=c_chan),
                           port_name_node1='portCA_1',
                           port_name_node2='portCB_1')

    return network


if __name__ == '__main__':

    # Repeat the protocol 5 times
    num_repeats = 5

    for _ in range(num_repeats):
        ns.sim_reset()

        n = generate_network()
        node_a = n.get_node("alice")
        node_b = n.get_node("bob")

        p1 = TeleportEncoderProtocol(node_a)
        p2 = TeleportDecoderProtocol(node_b)

        p1.start()
        p2.start()

        stats = ns.sim_run()
