import netsquid as ns
import numpy as np
import netsquid.components.instructions as instr
from netsquid.nodes import Node, Network, DirectConnection
from netsquid.components import QuantumChannel, QuantumProgram, ClassicalChannel, FibreDelayModel
from netsquid.protocols import NodeProtocol
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction


class EncodeQubitProgram(QuantumProgram):
    """
    Program to encode a bit according to a secret key and a basis.
    """

    default_num_qubits = 1

    def __init__(self, base, bit):
        super().__init__()
        self.base = base
        self.bit = bit

    def program(self):
        q1, = self.get_qubit_indices(1)
        self.apply(instr.INSTR_INIT, q1)
        if self.bit == 1:
            self.apply(instr.INSTR_X, q1)
        if self.base == 1:
            self.apply(instr.INSTR_H, q1)
        yield self.run()


class KeyReceiverProtocol(NodeProtocol):
    """
    Protocol for the receiver of the key.
    """

    def __init__(self, node, key_size=10, port_names=("portQB_1", "portCB_1", "portCB_2")):
        super().__init__(node)
        self.node = node
        self.q_port = port_names[0]
        self.c_port_i = port_names[1]
        self.c_port_o = port_names[2]
        self.key_size = key_size

    def run(self):
        # Select random bases
        bases = np.random.randint(2, size=self.key_size)
        results = []
        for i in range(self.key_size):
            # Await a qubit from Alice
            yield self.await_port_input(self.node.ports[self.q_port])

            # Measure in random basis
            if bases[i] == 0:
                res = self.node.qmemory.execute_instruction(instr.INSTR_MEASURE, output_key="M")
            else:
                res = self.node.qmemory.execute_instruction(instr.INSTR_MEASURE_X, output_key="M")
            yield self.await_program(self.node.qmemory)
            results.append(res[0]['M'][0])
            self.node.qmemory.reset()

            # Send ACK to Alice to trigger next qubit send (except in last transmit)
            if i < self.key_size - 1:
                self.node.ports[self.c_port_o].tx_output('ACK')

        # All qubits arrived, send bases
        self.node.ports[self.c_port_o].tx_output(bases)

        # Await matched indices from Alice and process key
        yield self.await_port_input(self.node.ports[self.c_port_i])
        matched_indices = self.node.ports[self.c_port_i].rx_input().items
        final_key = []
        for i in matched_indices:
            final_key.append(results[i])
        print('Bob\'s key', final_key)


class KeySenderProtocol(NodeProtocol):
    """
    Protocol for the sender of the key.
    """

    def __init__(self, node, key_size=10, port_names=("portQA_1", "portCA_1", "portCA_2")):
        super().__init__(node)
        self.node = node
        self.q_port = port_names[0]
        self.c_port_o = port_names[1]
        self.c_port_i = port_names[2]
        self.key_size = key_size

    def run(self):
        secret_key = np.random.randint(2, size=self.key_size)
        bases = list(np.random.randint(2, size=self.key_size))

        # Transmit encoded qubits to Bob
        for i, bit in enumerate(secret_key):
            self.node.qmemory.execute_program(EncodeQubitProgram(bases[i], bit))
            yield self.await_program(self.node.qmemory)

            q = self.node.qmemory.pop(0)
            self.node.ports[self.q_port].tx_output(q)
            if i < self.key_size - 1:
                yield self.await_port_input(self.node.ports[self.c_port_i])

        # Await response from Bob
        yield self.await_port_input(self.node.ports[self.c_port_i])
        bob_bases = self.node.ports[self.c_port_i].rx_input().items[0]
        matched_indices = []
        for i in range(self.key_size):
            if bob_bases[i] == bases[i]:
                matched_indices.append(i)

        self.node.ports[self.c_port_o].tx_output(matched_indices)
        final_key = []
        for i in matched_indices:
            final_key.append(secret_key[i])
        print('Alice\'s key', final_key)


def create_processor():
    """Factory to create a quantum processor for each end node.

    Has three memory positions and the physical instructions necessary
    for teleportation.
    """
    physical_instructions = [
        PhysicalInstruction(instr.INSTR_INIT, duration=3, parallel=True),
        PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True),
        PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True),
        PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False),
        PhysicalInstruction(instr.INSTR_MEASURE_X, duration=10, parallel=False)
    ]
    processor = QuantumProcessor("quantum_processor",
                                 phys_instructions=physical_instructions)
    return processor


def generate_network():
    """
    Generate the network. For BB84, we need a quantum and classical channel.
    """

    q_chan = QuantumChannel(name="AqB",
                            length=1,
                            models={"delay_model": FibreDelayModel()})

    c_chan_ab = ClassicalChannel(name="AcB",
                                 length=1,
                                 models={"delay_model": FibreDelayModel()})
    c_chan_ba = ClassicalChannel(name="BcA",
                                 length=1,
                                 models={"delay_model": FibreDelayModel()})

    network = Network("BB84 Network")
    alice = Node("alice", qmemory=create_processor())
    bob = Node("bob", qmemory=create_processor())

    network.add_nodes([alice, bob])
    _, p_ba = network.add_connection(alice,
                                     bob,
                                     label="q_chan",
                                     connection=DirectConnection(name="q_conn[A|B]",
                                                                 channel_AtoB=q_chan),
                                     port_name_node1="portQA_1",
                                     port_name_node2="portQB_1")
    # Map the qubit input port from the above channel to the memory index 0 on Bob"s
    # side
    bob.ports[p_ba].forward_input(bob.qmemory.ports["qin0"])
    network.add_connection(alice,
                           bob,
                           label="c_chan",
                           connection=DirectConnection(name="c_conn[A|B]",
                                                       channel_AtoB=c_chan_ab),
                           port_name_node1="portCA_1",
                           port_name_node2="portCB_1")
    network.add_connection(bob,
                           alice,
                           label="c_chan2",
                           connection=DirectConnection(name="c_conn[B|A]",
                                                       channel_AtoB=c_chan_ba),
                           port_name_node1="portCB_2",
                           port_name_node2="portCA_2")
    return network


if __name__ == '__main__':
    n = generate_network()
    node_a = n.get_node("alice")
    node_b = n.get_node("bob")

    p1 = KeySenderProtocol(node_a, key_size=15)
    p2 = KeyReceiverProtocol(node_b, key_size=15)

    p1.start()
    p2.start()

    stats = ns.sim_run()

