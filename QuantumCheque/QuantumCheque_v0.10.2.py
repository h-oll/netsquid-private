import netsquid as ns
import numpy as np
import netsquid.components.instructions as instr
from netsquid.nodes import Node, Network, DirectConnection
from netsquid.components import QuantumChannel, QuantumProgram, ClassicalChannel, FibreDelayModel
from netsquid.protocols import NodeProtocol, Signals
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from BB84 import KeyReceiverProtocol, KeySenderProtocol


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
        PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True),
        PhysicalInstruction(instr.INSTR_I, duration=2, parallel=True),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False),
        PhysicalInstruction(instr.INSTR_MEASURE_X, duration=10, parallel=False),
        PhysicalInstruction(instr.IGate(name='rotate_x'), duration=1, parallel=True),
        PhysicalInstruction(instr.IGate(name='rotate_y'), duration=1, parallel=True),
        PhysicalInstruction(instr.IGate(name='rotate_z'), duration=1, parallel=True),
    ]
    processor = QuantumProcessor("quantum_processor",
                                 num_positions=100,
                                 phys_instructions=physical_instructions)
    return processor


class GenerateGHZState(QuantumProgram):

    def __init__(self):
        super().__init__()

    def program(self):
        qs = self.get_qubit_indices(3)
        print(qs)
        self.apply(instr.INSTR_INIT, qs)
        self.apply(instr.INSTR_H, qs[0])
        self.apply(instr.INSTR_CNOT, [qs[0], qs[1]])
        self.apply(instr.INSTR_CNOT, [qs[0], qs[2]])
        yield self.run()


class TestProg(QuantumProgram):

    def __init__(self):
        super().__init__()

    def program(self):
        qs = self.get_qubit_indices(1)
        self.apply(instr.INSTR_H, qs[0])
        yield self.run()


class OneWayFunction(QuantumProgram):
    def __init__(self, secret_key, serial_number, salt, value, identity=0, primes=(33179, 32537, 31259)):
        super().__init__()
        self.primes = primes
        key = bin(identity)
        key += secret_key
        key += salt
        key += serial_number
        key += bin(value)[2:]
        self.key_value = int(key, 2)
        print(self.key_value)

    def program(self):
        def get_rx_matrix(rad):
            return [[np.cos(rad / 2), -1j * np.sin(rad / 2)], [-1j * np.sin(rad / 2), np.cos(rad / 2)]]

        def get_rz_matrix(rad):
            return [[np.exp(-1j * rad / 2), 0], [0, np.exp(1j * rad / 2)]]

        def get_ry_matrix(rad):
            return [[np.cos(rad / 2), -np.sin(rad / 2)], [np.sin(rad / 2), np.cos(rad / 2)]]

        angles = list(map(lambda x: np.pi / 180 * x, [self.key_value % self.primes[0],
                                                      self.key_value % self.primes[1],
                                                      self.key_value % self.primes[2]]))
        rx = ns.qubits.operators.Operator('rx', get_rx_matrix(angles[0]), cacheable=False)
        ry = ns.qubits.operators.Operator('ry', get_ry_matrix(angles[1]), cacheable=False)
        rz = ns.qubits.operators.Operator('rz', get_rz_matrix(angles[2]), cacheable=False)
        qs = self.get_qubit_indices(1)
        self.apply(instr.INSTR_H, qs)
        self.apply(instr.IGate(name='rotate_x', operator=rx), qs)
        self.apply(instr.IGate(name='rotate_y', operator=ry), qs)
        self.apply(instr.IGate(name='rotate_z', operator=rz), qs)
        yield self.run()


class CustomerProtocol(NodeProtocol):

    def __init__(self, node, value=100, port_names=("portQB_1", "portCB_1", "portCB_2"), n=3):
        super().__init__(node)
        self.node = node
        # Only 100 qubits allocated, can change this if memory size increases
        assert n <= 50
        self.n = n
        self.value = value
        self.q_port = port_names[0]
        self.c_port_i = port_names[1]
        self.c_port_o = port_names[2]
        qkd_rec = KeyReceiverProtocol(node, key_size=25, port_names=port_names)
        self.key_proto = qkd_rec
        self.add_subprotocol(qkd_rec, 'rec_key_proto')
        self.private_key = 'MIGrAgEAAiEAjfokFoJM3qwfye2XRvPpTuaeG7XtFDEubqw8Btf/lE8CAwEAAQIg' \
                           'SJEWyu50jceaQ+KNVLWshF2xm3CbMu+vFp9U5UD5T4ECEQD9K8+BsKGG1uGx2EC3' \
                           'WqRBAhEAj5BDRqhBlCPyRKuvZljUjwIRAOuz2Ro+LvQRpMhltELAUcECEQCBtLpD' \
                           'vUK6oBuD1YW8N2ebAhAA+wanWlg/3ZPdVa1aReWh'
        self.public_key = 'MCgCIQCN+iQWgkzerB/J7ZdG8+lO5p4bte0UMS5urDwG1/+UTwIDAQAB'
        self.shared_key = None

    def start(self):
        super().start()
        self.start_subprotocols()

    def run(self):
        # Get BB84 Key
        key_generated_signal = self.await_signal(
            sender=self.subprotocols['rec_key_proto'],
            signal_label=Signals.SUCCESS)
        yield key_generated_signal
        self.shared_key = self.key_proto.key

        # Send public key to bank
        self.node.ports[self.c_port_o].tx_output(self.public_key)

        # Receive GHZ states
        cur_port = 0
        for _ in range(self.n):
            for _ in range(2):
                self.node.ports[self.q_port].forward_input(self.node.qmemory.ports[f"qin{cur_port}"])
                yield self.await_port_input(self.node.ports[self.q_port])
                cur_port += 1
                self.node.ports[self.c_port_o].tx_output('ACK')


        # Qubits are in the memory as expected
        print('Customer memory', self.node.qmemory.peek(list(range(10))))
        q = self.node.qmemory.pop(0)[0]
        print(q.qstate)
        # Why does the qubit lose its state if I generate 2 GHZ groups????
        assert q.qstate is not None

        # Receive encrypted serial number of the cheque
        yield self.await_port_input(self.node.ports[self.c_port_i])
        encrypted_serial = self.node.ports[self.c_port_i].rx_input().items[0]
        # Decrypt serial number with a 1 time pad
        serial_number = []
        for i in range(len(encrypted_serial)):
            serial_number.append((int(encrypted_serial[i]) + self.shared_key[i]) % 2)

        # Apply the one way function
        str_shared_key = "".join([str(k) for k in self.shared_key])
        str_serial_num = "".join([str(s) for s in serial_number])

        # A simple test program with 1 gate to make sure qubits can be
        # operated on

        # self.node.qmemory.execute_program(TestProg(), qubit_mapping=[0])
        # yield self.await_program(self.node.qmemory)
        # print('did this')

        # for _ in range(self.n//3):
        #     mapping = 0
        #     r = np.random.randint(2)
        #     one_way_function = OneWayFunction(str_shared_key, str_serial_num, str(r), self.value)
        #     self.node.qmemory.execute_program(one_way_function, qubit_mapping=[mapping])
        #     yield self.await_program(self.node.qmemory)
        #     print('did this')
        #     mapping += 2


class BankerProtocol(NodeProtocol):

    def __init__(self, node, port_names=("portQA_1", "portCA_1", "portCA_2"), n=3):
        super().__init__(node)
        self.node = node
        # Only 100 qubits allocated, can change this if memory size increases
        assert n <= 50
        self.n = n
        self.q_port = port_names[0]
        self.c_port_i = port_names[2]
        self.c_port_o = port_names[1]
        qkd_sender = KeySenderProtocol(node, key_size=25, port_names=port_names)
        self.key_proto = qkd_sender
        self.shared_key = None
        self.add_subprotocol(qkd_sender, 'send_key_proto')

    def start(self):
        super().start()
        self.start_subprotocols()

    def run(self):
        key_generated_signal = self.await_signal(
            sender=self.subprotocols['send_key_proto'],
            signal_label=Signals.SUCCESS)
        yield key_generated_signal
        self.shared_key = self.key_proto.key
        yield self.await_port_input(self.node.ports[self.c_port_i])
        customer_public_key = self.node.ports[self.c_port_i].rx_input().items[0]

        # Generate n GHZ states and send 2 of the 3 to the customer, keeping the 1st
        mapping = (0, 1, 2)
        for _ in range(self.n):
            self.node.qmemory.execute_program(GenerateGHZState(), qubit_mapping=list(mapping))
            yield self.await_program(self.node.qmemory)
            qs = self.node.qmemory.pop(mapping[1:3])
            for q in qs:
                # Send qubit
                self.node.ports[self.q_port].tx_output(q)
                # Await ACK
                yield self.await_port_input(self.node.ports[self.c_port_i])
            mapping = tuple(map(lambda x: x + 3, mapping))

        # Bank should have n qubits remaining
        print('Bank memory', self.node.qmemory.peek(list(range(10))))

        # Send serial number with a 1 time pad
        serial_number = list(np.random.randint(2, size=len(self.shared_key)))
        encrypted_serial = []
        for i in range(len(serial_number)):
            encrypted_serial.append((serial_number[i] + self.shared_key[i]) % 2)
        t = "".join([str(s) for s in encrypted_serial])
        self.node.ports[self.c_port_o].tx_output(t)


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

    network = Network("Quantum Cheque Network")

    customer = Node("customer", qmemory=create_processor())
    banker = Node("banker", qmemory=create_processor())
    third_party = Node("third_party", qmemory=create_processor())

    network.add_nodes([customer, banker, third_party])
    _, p_ba = network.add_connection(banker,
                                     customer,
                                     label="q_chan",
                                     connection=DirectConnection(name="q_conn[A|B]",
                                                                 channel_AtoB=q_chan),
                                     port_name_node1="portQA_1",
                                     port_name_node2="portQB_1")
    customer.ports[p_ba].forward_input(customer.qmemory.ports["qin0"])
    network.add_connection(banker,
                           customer,
                           label="c_chan",
                           connection=DirectConnection(name="c_conn[A|B]",
                                                       channel_AtoB=c_chan_ab),
                           port_name_node1="portCA_1",
                           port_name_node2="portCB_1")
    network.add_connection(customer,
                           banker,
                           label="c_chan2",
                           connection=DirectConnection(name="c_conn[B|A]",
                                                       channel_AtoB=c_chan_ba),
                           port_name_node1="portCB_2",
                           port_name_node2="portCA_2")
    return network


if __name__ == '__main__':
    net = generate_network()
    node_a = net.get_node("customer")
    node_b = net.get_node("banker")

    p1 = CustomerProtocol(node_a, n=2)
    p2 = BankerProtocol(node_b, n=2)

    p1.start()
    p2.start()

    # ns.logger.setLevel(1)

    stats = ns.sim_run()
