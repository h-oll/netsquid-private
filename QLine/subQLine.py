'''
subQLine simulation 

Qubit sending direction
(A)------>(B)
This sub-portocol is made for QLine. 

Function:
Generate classical key between A and B. 
'''

import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.protocols import Protocol,LocalProtocol
from netsquid.qubits import create_qubits
from netsquid.qubits.operators import X,H,Z
from netsquid.nodes.connections import DirectConnection
from netsquid.components  import ClassicalFibre,QuantumFibre,QuantumMemory
from netsquid.components.models import  FibreDelayModel
from netsquid.pydynaa import Entity,EventHandler,EventType
from netsquid.protocols.protocol import Signals

from netsquid.qubits.qformalism import *
from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.components.models.qerrormodels import *

from random import randint

# get output value from a Qprogram
def getPGoutput(Pg,key):
    resList=[]
    tempDict=Pg.output
    value = tempDict.get(key, "")[0]
    return value



# A Quantum processor class which is able to send qubits from Qmemory
class sendableQProcessor(QuantumProcessor):
    def sendfromMem(self,inx,senderNode,senderPortName):
        payload=self.pop(inx)
        senderNode.ports[senderPortName].tx_output(payload)

        

# Quantum program operations on Node A 
class QG_A(QuantumProgram):
    def __init__(self,r,b):
        self.r=r
        self.b=b
        #print(self.r,self.b)
        super().__init__()
        
    def program(self):
        idx=self.get_qubit_indices(1)
        self.apply(INSTR_INIT, idx[0])
        if self.b==1:
            #print("did X")
            self.apply(INSTR_X, qubit_indices=idx[0])
        if self.r==1:
            #print("did H")
            self.apply(INSTR_H, qubit_indices=idx[0])
        
        yield self.run(parallel=False)



# Quantum program operations on Node B
class QG_B(QuantumProgram):
    def __init__(self,c,s):
        self.c=c
        self.s=s
        super().__init__()
        
    def program(self):
        idx=self.get_qubit_indices(1)
        if self.s==1:
            #print("did H")
            self.apply(INSTR_H,qubit_indices=idx[0])
        if self.c==1:
            #print("did X")
            self.apply(INSTR_X,qubit_indices=idx[0])
        self.apply(INSTR_MEASURE, qubit_indices=idx[0], output_key='R',physical=True) 
        
        yield self.run(parallel=False)
        
        

class subQLine(Protocol):
    
    
    def __init__(self
            ,node_A,processor_A
            ,node_B,processor_B
                 
            ,portNameQAB="portQAB"
            ,portNameCAB_1="portCAB_1",portNameCAB_2="portCAB_2"

            ,portNameQBA="portQBA"
            ,portNameCBA_1="portCBA_1",portNameCBA_2="portCBA_2"):

        self.qubitLoss=True # assume loss for init
        

        # init A
        self.node_A=node_A
        self.processor_A=processor_A
        self.r=randint(0,1)
        self.b=randint(0,1)
        self.portNameQAB=portNameQAB
        self.portNameCAB_1=portNameCAB_1
        self.portNameCAB_2=portNameCAB_2
        self.keyAB=[]
        
        self.node_A.ports[portNameCAB_1].bind_input_handler(self.A_compare)
        
        #self.node_A.ports[portNameQA2].bind_input_handler(self.A_receive) 
        
        
        # init B
        self.node_B=node_B
        self.processor_B=processor_B
        self.c=randint(0,1)
        self.s=randint(0,1)
        self.portNameQBA=portNameQBA
        self.portNameCBA_1=portNameCBA_1
        self.portNameCBA_2=portNameCBA_2
        self.keyBA=[]
        
        #print("========================================")
        self.node_B.ports[portNameQBA].bind_input_handler(self.B_receive_prepare)
        self.node_B.ports[portNameCBA_2].bind_input_handler(self.B_compare)
        
        
        self.TimeF=0.0
        self.TimeD=0.0
        
        super().__init__()
        
        
    def start(self):
        super().start()
        self.TimeF=ns.util.simtools.sim_time(magnitude=ns.NANOSECOND) #in nanoseconds
        self.A_prepare()
        

        
# 1 =====================================================================        
    def A_prepare(self):
        myQG_A=QG_A(self.r,self.b)
        self.processor_A.execute_program(myQG_A,qubit_mapping=[0])
        self.processor_A.set_program_done_callback(self.A_send,once=True)
        self.processor_A.set_program_fail_callback(self.A_Fail,once=True)

    
    def A_Fail(self):
        self.qubitLoss=True
        print("A failed Qprogram!")
        
        
    def A_send(self):
        self.processor_A.sendfromMem(senderNode=self.node_A
            ,inx=[0]                  # get even index in qList
            ,senderPortName=self.node_A.ports[self.portNameQAB].name)
        

# 2  =====================================================================  
    def B_receive_prepare(self,qubit):
        #print("B received:",qubit.items)

        self.processor_B.put(qubits=qubit.items)

        self.myQG_B=QG_B(self.c,self.s)
        self.processor_B.execute_program(self.myQG_B,qubit_mapping=[0])
        self.processor_B.set_program_done_callback(self.B_send,once=True)
        self.processor_B.set_program_fail_callback(self.B_PgFail,once=True)

        
    def B_PgFail(self):
        self.qubitLoss=True
        print("B Qprogrm failed!")
        
 

    def B_send(self):
        self.R=getPGoutput(self.myQG_B,'R')
        
        self.node_B.ports[self.portNameCBA_1].tx_output([self.s,self.R])
        


# 3 =====================================================================
        
    
    def A_compare(self,alist): # first in the list is s, second is R
        if alist.items[0]==self.r: # if s == r
            self.keyAB.append(self.b)
        self.A_announce_B()
        

    def A_announce_B(self):
        self.node_A.ports[self.portNameCAB_2].tx_output(self.r)
    
    
# 4 =====================================================================

    def B_compare(self,r):
        self.qubitLoss=False
        if self.s==r.items[0]:
            self.keyBA.append(self.R ^ self.c)
        self.TimeD=ns.util.simtools.sim_time(magnitude=ns.NANOSECOND)-self.TimeF #in nanoseconds
        
        

def run_QLine_sim(times=1,fibre_len=10**-3,noise_model=None): # fibre 1 m long
    QlineKeyListA=[]
    QlineKeyListB=[]
    totalTime=0.0
    qubitLossCount=0
    #TimeF=ns.util.simtools.sim_time(magnitude=ns.NANOSECOND)
    #DepolarNoiseModel
    
    
    # AB Hardware configuration
    processorA=sendableQProcessor("processor_A", num_positions=1,
                    mem_noise_models=None, phys_instructions=[
                    PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
                    PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True)],
                    topologies=[None, None, None, None])

    #mem_noise_models=[DepolarNoiseModel(0)] * 100
    processorB=sendableQProcessor("processor_B", num_positions=1,
                mem_noise_models=None, phys_instructions=[
                PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
                PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
                PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
                PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True)],
                topologies=[None, None, None, None])


    node_A = Node("A",ID=0,port_names=["portQAB","portCAB_1","portCAB_2"])
    node_B = Node("B",ID=1,port_names=["portQBA","portCBA_1","portCBA_2"])#



    # Quantum fibre
    MyQfiberAB=QuantumFibre("QFibre_A->B", length=fibre_len
        ,p_loss_init=0.04, p_loss_length=0.25)#,quantum_loss_model="default"
    # default value:
    # p_loss_init=0.2, p_loss_length=0.25,depolar_rate=0, c=200000, models=None


    # Classical fibre
    MyCfiberBA=DirectConnection("CFibreConn_B->A",
        ClassicalFibre("CFibre_B->A", length=fibre_len))
    MyCfiberAB=DirectConnection("CFibreConn_A->B",
        ClassicalFibre("CFibre_A->B", length=fibre_len))

    # Quantum connection
    node_A.connect_to(node_B, MyQfiberAB,
        local_port_name =node_A.ports["portQAB"].name,
        remote_port_name=node_B.ports["portQBA"].name)

    # Classical connection
    node_B.connect_to(node_A, MyCfiberBA,
        local_port_name="portCBA_1", remote_port_name="portCAB_1")
    node_A.connect_to(node_B, MyCfiberAB,
        local_port_name="portCAB_2", remote_port_name="portCBA_2")

    
    for i in range(times):
        ns.sim_reset()

        myQLine=subQLine(node_A=node_A,processor_A=processorA
            ,node_B=node_B,processor_B=processorB
            ,portNameQAB="portQAB"
            ,portNameCAB_1="portCAB_1",portNameCAB_2="portCAB_2"

            ,portNameQBA="portQBA"
            ,portNameCBA_1="portCBA_1",portNameCBA_2="portCBA_2")
        
        myQLine.start()
        ns.sim_run()
        
        if myQLine.keyAB and myQLine.keyAB==myQLine.keyBA:
            QlineKeyListA.append(myQLine.keyAB[0])
            QlineKeyListB.append(myQLine.keyBA[0])
            totalTime+=myQLine.TimeD
            
        if myQLine.qubitLoss==True:
            qubitLossCount+=1
            
            
    print("first key:",QlineKeyListA)
    
    '''
    # second key gen=================================================================
    #BC hardware configuration
    
    
    processorC=sendableQProcessor("processor_C", num_positions=1,
                    mem_noise_models=None, phys_instructions=[
                    PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
                    PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True)],
                    topologies=[None, None, None, None])


    node_C = Node("C",ID=2,port_names=["portQCB","portCCB_1","portCCB_2"])
    node_B.add_ports("portQBC")
    
    # Quantum fibre
    MyQfiberBC=QuantumFibre("QFibre_B->C", length=fibre_len
            ,p_loss_init=0.04, p_loss_length=0.25)
    # Quantum connection
    node_B.connect_to(node_C, MyQfiberBC,
            local_port_name =node_B.ports["portQBC"].name,
            remote_port_name=node_C.ports["portQCB"].name)

        
    
    
    # Classical fibre
    MyCfiberCB=DirectConnection("CFibreConn_C->B",
            ClassicalFibre("CFibre_C->B", length=fibre_len))
    MyCfiberBC=DirectConnection("CFibreConn_B->C",
            ClassicalFibre("CFibre_B->C", length=fibre_len))
    # Classical connection
    node_B.connect_to(node_C, MyCfiberBC,
            local_port_name="portCBC_1", remote_port_name="portCCB_1")
    node_C.connect_to(node_B, MyCfiberCB,
            local_port_name="portCCB_2", remote_port_name="portCBC_2")
    
    
    
    # Port Forwarding
    node_B.ports["portQBA"].forward_input(node_C.ports["portQCB"])
    node_C.ports["portCCB_1"].forward_input(node_B.ports["portCBA_1"])
    node_B.ports["portCBA_2"].forward_input(node_C.ports["portCCB_2"])
    
    

    for i in range(times):
        ns.sim_reset()
        myQLine2=subQLine(node_A=node_A,processor_A=processorA
            ,node_B=node_C,processor_B=processorC
            ,portNameQAB="portQAB"
            ,portNameCAB_1="portCAB_1",portNameCAB_2="portCAB_2"

            ,portNameQBA="portQBA"
            ,portNameCBA_1="portCBA_1",portNameCBA_2="portCBA_2")
        
        
        myQLine2.start()
        ns.sim_run()
    

        if myQLine2.keyAB and myQLine.keyAB==myQLine.keyBA:
            QlineKeyListA.append(myQLine.keyAB[0])
            QlineKeyListB.append(myQLine.keyBA[0])
            totalTime+=myQLine.TimeD
            
        if myQLine2.qubitLoss==True:
            qubitLossCount+=1
            
            
    print("second key:",QlineKeyListA) 
    
        

    
    if totalTime !=0:
        #key length per nanosec, proportion of qubit loss
        return len(QlineKeyListA)/totalTime
    else :
        return 0   #, qubitLossCount/times
    
    
    if times != 0:
        return qubitLossCount/times
    '''

# test
run_QLine_sim(times=20,fibre_len=10**-3,noise_model=None) #DepolarNoiseModel(depolar_rate=500)


