import netsquid as ns
import numpy as np
import netsquid.components.instructions as instr
from netsquid.nodes import Node, Network, DirectConnection
from netsquid.components import QuantumChannel, QuantumProgram, ClassicalChannel, FibreDelayModel
from netsquid.protocols import NodeProtocol, Signals
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from BB84_v0_10_2 import KeyReceiverProtocol, KeySenderProtocol


def create_processor():
    """Factory to create a quantum processor for each end node.

    Has three memory positions and the physical instructions necessary
    for teleportation.
    """
    physical_instructions = [
        PhysicalInstruction(instr.INSTR_INIT, duration=3, parallel=False),
        PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True),
        PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True),
        PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True),
        PhysicalInstruction(instr.INSTR_Y, duration=1, parallel=True),
        PhysicalInstruction(instr.INSTR_T, duration=1, parallel=True),
        PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True),
        PhysicalInstruction(instr.INSTR_I, duration=2, parallel=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False),
        PhysicalInstruction(instr.INSTR_MEASURE_X, duration=10, parallel=False),
        PhysicalInstruction(instr.IGate(name='CSWAP'), duration=8, parallel=True),
        PhysicalInstruction(instr.IGate(name='rotate_x'), duration=1, parallel=True),
        PhysicalInstruction(instr.IGate(name='rotate_y'), duration=1, parallel=True),
        PhysicalInstruction(instr.IGate(name='rotate_z'), duration=1, parallel=True),
    ]
    processor = QuantumProcessor("quantum_processor",
                                 num_positions=5,
                                 phys_instructions=physical_instructions)
    return processor


class GenerateGHZState(QuantumProgram):

    def __init__(self):
        super().__init__()

    def program(self):
        qs = self.get_qubit_indices(3)
        self.apply(instr.INSTR_INIT, qs)
        self.apply(instr.INSTR_H, qs[0])
        self.apply(instr.INSTR_CNOT, [qs[0], qs[1]])
        self.apply(instr.INSTR_CNOT, [qs[0], qs[2]])
        yield self.run()


class SwapTest(QuantumProgram):

    def __init__(self):
        super().__init__()

    def program(self):
        qs = self.get_qubit_indices(3)
        self.apply(instr.INSTR_INIT, qs[0])
        self.apply(instr.INSTR_H, qs[0])
        cswap_matrix = [[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]]
        cswap = ns.qubits.operators.Operator('cswap', cswap_matrix, cacheable=True)
        self.apply(instr.IGate(name='CSWAP', operator=cswap), qs)
        self.apply(instr.INSTR_H, qs[0])
        self.apply(instr.INSTR_MEASURE, qs[0], output_key="M")
        yield self.run()


class OneWayFunction(QuantumProgram):
    def __init__(self, secret_key, serial_number, salt, value, identity=0, primes=(33179, 32537, 31259)):
        super().__init__()
        self.primes = primes
        key = str(bin(identity))
        key += secret_key
        key += salt
        key += serial_number
        key += bin(value)[2:]
        self.key_value = int(key, 2)

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
        q, = self.get_qubit_indices(1)
        self.apply(instr.INSTR_INIT, q)
        self.apply(instr.IGate(name='rotate_x', operator=rx), q)
        self.apply(instr.IGate(name='rotate_y', operator=ry), q)
        self.apply(instr.IGate(name='rotate_z', operator=rz), q)
        yield self.run()


class CustomerProtocol(NodeProtocol):

    def __init__(self, node, value=2200, port_names=("qubitIO", "classicIO"), n=30):
        super().__init__(node)
        self.node = node
        self.n = n
        self.value = value
        self.q_port = port_names[0]
        self.c_port = port_names[1]
        qkd_rec = KeyReceiverProtocol(node, key_size=5 * n, port_names=port_names)
        self.key_proto = qkd_rec
        self.add_subprotocol(qkd_rec, 'rec_key_proto')
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
        self.node.qmemory.reset()

        self.node.ports[self.c_port].tx_output('START SERIAL')

        # Receive encrypted serial number of the cheque
        yield self.await_port_input(self.node.ports[self.c_port])
        encrypted_serial = self.node.ports[self.c_port].rx_input().items[0]
        # Decrypt serial number with a 1 time pad
        serial_number = [(int(encrypted_serial[i]) + self.shared_key[i]) % 2 for i in range(len(encrypted_serial))]

        str_shared_key = "".join([str(k) for k in self.shared_key])
        str_serial_num = "".join([str(s) for s in serial_number])
        random_bits = np.random.randint(2, size=self.n)

        # Send random string to Bank
        encoded_random = []
        for i in range(len(random_bits)):
            encoded_random.append((int(random_bits[i]) + self.shared_key[i]) % 2)
        self.node.ports[self.c_port].tx_output({'value': self.value, 'encoded_bits': encoded_random})

        for i in range(self.n):
            # Receive GHZ states
            for p in range(2):
                self.node.ports[self.q_port].forward_input(self.node.qmemory.ports[f"qin{p}"])
                yield self.await_port_input(self.node.ports[self.q_port])
                self.node.ports[self.c_port].tx_output('ACK')

            # Encode qubits
            one_way_function = OneWayFunction(str_shared_key, str_serial_num, str(random_bits[i]), self.value)
            self.node.qmemory.execute_program(one_way_function, qubit_mapping=[2])
            yield self.await_program(self.node.qmemory)

            # Entangle qubits
            self.node.qmemory.execute_instruction(instr.INSTR_CNOT, qubit_mapping=[2, 0])
            yield self.await_program(self.node.qmemory)

            res1 = self.node.qmemory.execute_instruction(instr.INSTR_MEASURE, qubit_mapping=[0],
                                                         output_key="M")
            yield self.await_program(self.node.qmemory)
            res2 = self.node.qmemory.execute_instruction(instr.INSTR_MEASURE_X, qubit_mapping=[2],
                                                         output_key="M")
            yield self.await_program(self.node.qmemory)

            res1, res2 = res1[0]['M'][0], res2[0]['M'][0]
            if res1 == 0 and res2 == 1:
                self.node.qmemory.execute_instruction(instr.INSTR_Z, qubit_mapping=[1])
                yield self.await_program(self.node.qmemory)
            elif res1 == 1 and res2 == 0:
                self.node.qmemory.execute_instruction(instr.INSTR_X, qubit_mapping=[1])
                yield self.await_program(self.node.qmemory)
            elif res1 == 1 and res2 == 1:
                self.node.qmemory.execute_instruction(instr.INSTR_Y, qubit_mapping=[1])
                yield self.await_program(self.node.qmemory)

            q = self.node.qmemory.pop(1)
            self.node.ports[self.q_port].tx_output(q)
            # Await ACK
            yield self.await_port_input(self.node.ports[self.c_port])

        yield self.await_port_input(self.node.ports[self.c_port])
        cheque_valid = self.node.ports[self.c_port].rx_input().items[0]
        print('The cheque was:', cheque_valid)


class BankerProtocol(NodeProtocol):

    def __init__(self, node, port_names=("qubitIO", "classicIO"), n=30):
        super().__init__(node)
        self.node = node
        self.n = n
        self.q_port = port_names[0]
        self.c_port = port_names[1]
        qkd_sender = KeySenderProtocol(node, key_size=5 * n, port_names=port_names)
        self.key_proto = qkd_sender
        self.shared_key = None
        self.add_subprotocol(qkd_sender, 'send_key_proto')

    def start(self):
        super().start()
        self.start_subprotocols()

    def run(self):
        # Send BB84 Key
        key_generated_signal = self.await_signal(
            sender=self.subprotocols['send_key_proto'],
            signal_label=Signals.SUCCESS)
        yield key_generated_signal
        self.shared_key = self.key_proto.key
        self.node.qmemory.reset()

        # Await start trigger for serial
        yield self.await_port_input(self.node.ports[self.c_port])

        # Send serial number with a 1 time pad
        serial_number = list(np.random.randint(2, size=len(self.shared_key)))
        encrypted_serial = [(serial_number[i] + self.shared_key[i]) % 2 for i in range(len(serial_number))]
        t = "".join([str(s) for s in encrypted_serial])
        self.node.ports[self.c_port].tx_output(t)

        # Await encrypted random string for customer and cheque value
        yield self.await_port_input(self.node.ports[self.c_port])
        cheque_info = self.node.ports[self.c_port].rx_input().items[0]
        customer_bits = [(a + self.shared_key[i]) % 2 for (i, a) in enumerate(cheque_info['encoded_bits'])]
        value = cheque_info['value']

        # For regenerating signature
        str_shared_key = "".join([str(k) for k in self.shared_key])
        str_serial_num = "".join([str(s) for s in serial_number])

        self.node.ports[self.q_port].forward_input(self.node.qmemory.ports["qin1"])
        swap_test_results = []
        for i in range(self.n):
            self.node.qmemory.execute_program(GenerateGHZState())
            yield self.await_program(self.node.qmemory)
            qs = self.node.qmemory.pop([1, 2])
            for q in qs:
                self.node.ports[self.q_port].tx_output(q)
                # Await ACK
                yield self.await_port_input(self.node.ports[self.c_port])

            # Await cheque qubit
            yield self.await_port_input(self.node.ports[self.q_port])
            self.node.ports[self.c_port].tx_output('ACK')

            # Perform Corrections
            m = self.node.qmemory.execute_instruction(instr.INSTR_MEASURE_X, qubit_mapping=[0], output_key="M")
            yield self.await_program(self.node.qmemory)
            if m[0]['M'][0] == 1:
                self.node.qmemory.execute_instruction(instr.INSTR_Z, qubit_mapping=[1])
                yield self.await_program(self.node.qmemory)

            one_way_function = OneWayFunction(str_shared_key, str_serial_num, str(customer_bits[i]), value)
            self.node.qmemory.execute_program(one_way_function, qubit_mapping=[2])
            yield self.await_program(self.node.qmemory)
            # Perform swap test
            swap_test = SwapTest()
            self.node.qmemory.execute_program(swap_test, qubit_mapping=[0, 1, 2])
            yield self.await_program(self.node.qmemory)
            swap_test_results.append(swap_test.output['M'][0])

        corr_percent = 100 * (1 - sum(swap_test_results) / self.n)
        threshold_constant_percent = (1 - (3 / 4) ** self.n) * 100
        print('Correlation was:', corr_percent, 'Threshold is:', threshold_constant_percent)
        if corr_percent >= threshold_constant_percent:
            print('Cheque was accepted')
        else:
            print('Cheque was rejected')


def generate_network():
    """
    Generate the network. For BB84, we need a quantum and classical channel.
    """

    network = Network("Quantum Cheque Network")

    customer = Node("customer", qmemory=create_processor())
    banker = Node("banker", qmemory=create_processor())

    network.add_nodes([customer, banker])
    p_ab, p_ba = network.add_connection(banker,
                                        customer,
                                        label="q_chan",
                                        channel_to=QuantumChannel('AqB', delay=10),
                                        channel_from=QuantumChannel('BqA', delay=10),
                                        port_name_node1="qubitIO",
                                        port_name_node2="qubitIO")
    customer.ports[p_ba].forward_input(customer.qmemory.ports["qin0"])
    banker.ports[p_ab].forward_input(banker.qmemory.ports["qin0"])
    network.add_connection(banker,
                           customer,
                           label="c_chan",
                           channel_to=ClassicalChannel('AcB', delay=10),
                           channel_from=ClassicalChannel('BcA', delay=10),
                           port_name_node1="classicIO",
                           port_name_node2="classicIO")
    return network


if __name__ == '__main__':
    net = generate_network()
    node_a = net.get_node("customer")
    node_b = net.get_node("banker")

    p1 = CustomerProtocol(node_a, n=55)
    p2 = BankerProtocol(node_b, n=55)

    p1.start()
    p2.start()

    # ns.logger.setLevel(1)

    stats = ns.sim_run()
