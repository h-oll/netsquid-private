

import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from BB84_lib_v1_5 import *
from netsquid.qubits import ketstates
from netsquid.qubits.operators import *
from netsquid.qubits.qubitapi import *
from netsquid.qubits import operators as ops

from random import seed, randint





def Create_GHZ_qubits(num_qubits=8):
    qList = create_qubits(num_qubits,system_name="Q") #qubit 00
    ns.qubits.operate(qList[0], ns.H)
    ns.qubits.operate([qList[0],qList[1]], ns.CNOT) 
    for i in range(2,num_qubits):
        ns.qubits.operate([qList[0],qList[i]], ns.CNOT)
    return qList



def RandomSerialNumGen(num_bits):
    seed(randint(0, 2**num_bits))
    bitList=[]
    for i in range(num_bits):
        bitList.append(randint(0,1))
    return bitList



# hash with the symmetric key, random Serial Number, the amoud of money
def OneWayFunction(identity=None,symkey=[],randomSerialNumber=0,Money=0):
    owf_key=''
    for i in symkey:
        owf_key+=str(bin(i)[2:])
        
    owf_key+=str(bin(randomSerialNumber)[2:])
    owf_key+=str(bin(Money)[2:]) 
    owf_key=int(owf_key)
    
    # make it qubit
    # apply three prime numbers
    p1 = 33179
    p2 = 32537
    p3 = 31259
    
    MyRx=create_rotation_op(np.pi/180*(owf_key%p1), (1, 0, 0))
    MyRy=create_rotation_op(np.pi/180*(owf_key%p2), (0, 1, 0))
    MyRz=create_rotation_op(np.pi/180*(owf_key%p3), (0, 0, 1))
    
    tempQubit=create_qubits(1)
    tempQubit=tempQubit[0]
    MyRx | tempQubit
    MyRy | tempQubit
    MyRz | tempQubit
    
    #print(tempQubit.qstate.dm)
    return tempQubit
    
# C swap can be composed by T
# https://www.mathstat.dal.ca/~selinger/quipper/doc/QuipperLib-GateDecompositions.html
def Cswap(qA,qB,qC):
    
    invT=T.inv
    
    operate([qC, qB], ops.CNOT)
    H | qC
    T | qA
    T | qB
    T | qC
    operate([qB, qA], ops.CNOT)
    operate([qC, qB], ops.CNOT)
    operate([qA, qC], ops.CNOT)
    T | qC
    invT | qB
    operate([qA, qB], ops.CNOT)
    invT | qA
    invT | qB
    operate([qC, qB], ops.CNOT)
    operate([qA, qC], ops.CNOT)
    operate([qB, qA], ops.CNOT)
    H | qC
    operate([qC, qB], ops.CNOT)
    
    return qA,qB,qC



def SwapTest(qB,qC):
    qA=create_qubits(1)
    qA=qA[0]

    H | qA
    Cswap(qA,qB,qC)
    H | qA

    return ns.qubits.qubitapi.measure(qA,observable=Z)
    





class QuantumCheque(Protocol):
  
    # QuantumCheque function =========================================================

    def B_send_checkbook(self):
        print("B_send_checkbook called")
        self.BChequeBook=Create_GHZ_qubits(self.num_bits)
        # save one qubit for bank
        self.saveChequeBook=self.BChequeBook[0]
        # send others to A
        self.node_B.ports["portQ_B1"].tx_output(self.BChequeBook[1:])
    
    
    
    def A_RSG_send_B(self,qList):
        print("A_RSG_send_B called")
        self.AChequeBook=qList.items
        self.A_saveRSB=RandomSerialNumGen(self.num_bits)
        self.node_A.ports["portC_A1"].tx_output(self.A_saveRSB)
    
   

    def BA_BB84_keygen(self,RSB,num_bits=8,fibre_len=10**-6,
                 fibre_loss_init=0,fibre_loss_length=0):
        print("BA_BB84_keygen")
        self.B_saveRSB=RSB
        loc_BB84=BB84(num_bits=num_bits,fibre_len=fibre_len)
        self.key_BB84_A=loc_BB84.key_A
        self.key_BB84_B=loc_BB84.key_B
        # trigger A's one way function
        self.node_B.ports["portC_B2"].tx_output('Bob')
        
        
        
    def A_owf_send_C(self,identity):
        identity=identity.items[0]
        
        if str(identity)=='Bob':
            res_owf_qubit=OneWayFunction(identity,self.key_BB84_A,self.A_saveRSB.pop(),456)

        
            if len(self.AChequeBook)<=0:
                print("No more cheque! Aborting!")
                return 0
            else:
                # Alice performs Bell state measurement
                operate([res_owf_qubit, self.AChequeBook.pop()], ops.CNOT)
                H | res_owf_qubit
            
                # not done
                #
                #
            
            
            
            self.node_A.ports["portQ_A2"].tx_output(res_owf_qubit)
        else:
            print("Error! message not received from Bob!")
            

    # a list of qubits as Qcheque is ready to be verified by bank
    def CB_Verify(self,chequeQ):
        print("CB_Verify called")
        chequeQ=chequeQ.items
        print(chequeQ[0].qstate.dm)
        
        
    
    
    # base function ========================================================
    
    def __init__(self, num_bits=8,fibre_len_AC=10**-6
                 ,fibre_len_AB=10**-6,fibre_len_BC=10**-6): 
        super().__init__()
        self.node_A = Node("A",ID=0,port_names=["portQ_A1","portC_A1","portC_A2","portQ_A2"])
        self.node_B = Node("B",ID=1,port_names=["portQ_B1","portC_B1","portC_B2","portQ_B2"])
        self.node_C = Node("C",ID=2,port_names=["portQ_C1","portQ_C2"])
        
        self.MyQfiber_AC = None
        self.MyQfiber_AB = None
        self.MyQfiber_BC = None
        
        self.MyCfiber_AB = None
        self.MyCfiber_BA = None
        
        self.fiberLenth_AC = fibre_len_AC
        self.fiberLenth_AB = fibre_len_AB
        self.fiberLenth_BC = fibre_len_BC
        
        self.num_bits = num_bits
        self.AChequeBook = [] # qubits of self.num_bits-1
        self.BChequeBook = [] # one qubit
        self.saveChequeBook = None

        self.A_saveRSB=[] #random serial bits stored in A
        self.B_saveRSB=[] #random serial bits stored in B
        
        self.key_BB84_A = []
        self.key_BB84_B = []
        
        self.start()
        
        
    def stop(self):
        super().stop()
        self._running = False
        
        
    def is_connected():
        super().is_connected()
        pass
        
        
    def start(self):
        super().start()
        
        self.MyCfiber_AB=DirectConnection("CFibreConn_AB",
                ClassicalFibre("CFibre_A->B", length=self.fiberLenth_AB),
                ClassicalFibre("CFibre_B->A", length=self.fiberLenth_AB))
        self.MyCfiber_BA=DirectConnection("CFibreConn_BA",
                ClassicalFibre("CFibre_A->B", length=self.fiberLenth_AB),
                ClassicalFibre("CFibre_B->A", length=self.fiberLenth_AB))
        
        
        self.MyQfiber_AC=QuantumFibre("MyQFibre_AC", length=self.fiberLenth_AC, 
            loss_model="default")
        self.MyQfiber_AB=QuantumFibre("MyQFibre_AB", length=self.fiberLenth_AB, 
            loss_model="default")
        self.MyQfiber_BC=QuantumFibre("MyQFibre_BC", length=self.fiberLenth_BC, 
            loss_model="default")
        
        self.node_B.connect_to(self.node_A, self.MyQfiber_AB,
            local_port_name="portQ_B1", remote_port_name="portQ_A1")
        self.node_A.connect_to(self.node_B, self.MyCfiber_AB,
            local_port_name="portC_A1", remote_port_name="portC_B1")
        self.node_B.connect_to(self.node_A, self.MyCfiber_BA,
            local_port_name="portC_B2", remote_port_name="portC_A2")
        self.node_A.connect_to(self.node_C, self.MyQfiber_AC,
            local_port_name="portQ_A2", remote_port_name="portQ_C1")
        self.node_C.connect_to(self.node_B, self.MyQfiber_BC,
            local_port_name="portQ_C2", remote_port_name="portQ_B2")
        
        
        
        # set callback functions===================================================
        self.node_A.ports["portQ_A1"].bind_input_handler(self.A_RSG_send_B)
        self.node_B.ports["portC_B1"].bind_input_handler(self.BA_BB84_keygen)
        self.node_A.ports["portC_A2"].bind_input_handler(self.A_owf_send_C)
        self.node_C.ports["portQ_C1"].bind_input_handler(self.CB_Verify)
        
        
        # Start by bank
        self.B_send_checkbook()
        
        
        



ns.sim_reset()
qc=QuantumCheque(10)
ns.sim_run()


