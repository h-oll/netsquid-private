#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


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
    
        yield self.run(parallel=False)


# Quantum program operations on Node C
class QG_C(QuantumProgram):
    def __init__(self):
        super().__init__()
        
    def program(self):
        self.apply(INSTR_MEASURE, qubit_indices=0, output_key='r',physical=True) 
    
        yield self.run(parallel=False)


# In[5]:


class QLine(Protocol):
    
    def __init__(self
            ,node_A,processor_A
            ,node_B,processor_B
            ,node_C,processor_C
            ,portNameQA1="portQA_1"
            ,portNameCA1="portCA_1",portNameCA2="portCA_2"

            ,portNameQB1="portQB_1",portNameQB2="portQB_2"
            ,portNameCB1="portCB_1",portNameCB2="portCB_2",portNameCB3="portCB_3"

            ,portNameQC1="portQC_1",portNameCC1="portCC_1"):

        
        # init A
        self.node_A=node_A
        self.processor_A=processor_A
        self.r=randint(0,1)
        self.b=randint(0,1)
        self.portNameQA1=portNameQA1
        self.portNameCA1=portNameCA1
        self.portNameCA2=portNameCA2
        self.keyA=[]
        
        self.node_A.ports[portNameCA1].bind_input_handler(self.A_compare)
        
        
        
        # init B
        self.node_B=node_B
        self.processor_B=processor_B
        self.c=randint(0,1)
        self.s=randint(0,1)
        self.portNameQB1=portNameQB1
        self.portNameQB2=portNameQB2
        self.portNameCB1=portNameCB1
        self.portNameCB2=portNameCB2
        self.portNameCB3=portNameCB3
        self.keyB=[]
        
        self.node_B.ports[portNameQB1].bind_input_handler(self.B_receive_prepare)
        self.node_B.ports[portNameCB3].bind_input_handler(self.B_ack_send_A)
        self.node_B.ports[portNameCB2].bind_input_handler(self.B_compare)
        
        # init C
        self.node_C=node_C
        self.processor_C=processor_C
        self.portNameQC1=portNameQC1
        self.portNameCC1=portNameCC1
        self.R=None
        
        self.node_C.ports[portNameQC1].bind_input_handler(self.C_measure)
        
        super().__init__()
        
        
        
    def start(self):
        super().start()
        self.A_prepare()
        
# 1 =====================================================================        
    def A_prepare(self):
        myQG_A=QG_A(self.r,self.b)
        self.processor_A.execute_program(myQG_A,qubit_mapping=[0])
        self.processor_A.set_program_done_callback(self.A_send,once=True)
        self.processor_A.set_program_fail_callback(self.A_Fail,once=True)
    
    
    def A_Fail(self):
        print("A failed Qprogram!")
        
        
    def A_send(self):
        self.processor_A.sendfromMem(senderNode=self.node_A
            ,inx=[0]                  # get even index in qList
            ,senderPortName=self.node_A.ports[self.portNameQA1].name)
        
    
# 2  =====================================================================  
    def B_receive_prepare(self,qubit):
        #print("received:",qubit.items)
        #print("B's  c,s: ",self.c,self.s)
        
        self.processor_B.put(qubits=qubit.items)
        
        myQG_B=QG_B(self.c,self.s)
        self.processor_B.execute_program(myQG_B,qubit_mapping=[0])
        self.processor_B.set_program_done_callback(self.B_send,once=True)
        self.processor_B.set_program_fail_callback(self.B_PgFail,once=True)
        
        
    def B_PgFail(self):
        print("B Qprogrm failed!")
        
 

    def B_send(self):
        self.processor_B.sendfromMem(senderNode=self.node_B
            ,inx=0
            ,senderPortName=self.node_B.ports[self.portNameQB2].name)
        


# 3 =====================================================================
    def C_measure(self,qubit):
        #print("C_measure received:",qubit.items)
        self.processor_C.put(qubits=qubit.items)
        self.myQG_C=QG_C()
        self.processor_C.execute_program(self.myQG_C,qubit_mapping=[0])
        self.processor_C.set_program_done_callback(self.C_send,once=True)
        self.processor_C.set_program_fail_callback(self.C_PgFail,once=True)
        
        
        
    def C_PgFail(self):
        print("C Qprogram failed!")        
        

    # C send classical data to B
    def C_send(self):
        self.R=getPGoutput(self.myQG_C,'r')
        if self.R != None:
            self.node_C.ports[self.portNameCC1].tx_output(self.R)
        else:
            print("Error! R is None")
        
# 4 =====================================================================
        
    def B_ack_send_A(self,R):
        self.node_B.ports[self.portNameCB1].tx_output([R.items[0],self.s])
    
    
# 5 =====================================================================
    
    def A_compare(self,alist): # first in the list is R, second is s
        if alist.items[1]==self.r: # if s == r
            self.keyA.append(self.b)
        self.A_announce_B()
        

    def A_announce_B(self):
        self.node_A.ports[self.portNameCA2].tx_output(self.r)
    
    
# 6 =====================================================================

    def B_compare(self,r):
        if self.s==r.items[0]:
            self.keyB.append(self.R ^ self.c)

 


# In[27]:


def run_QLine_sim(times=1,fibre_len=0,noise_model=None):
    QlineKeyListA=[]
    QlineKeyListB=[]
    
    for i in range(times):
        ns.sim_reset()

        # Hardware configuration
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

        processorC=sendableQProcessor("processor_C", num_positions=1,
                    mem_noise_models=None, phys_instructions=[
                    PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
                    PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True)],
                    topologies=[None, None, None, None])


        node_A = Node("A",ID=0,port_names=["portQA_1","portCA_1","portCA_2"])
        node_B = Node("B",ID=1,port_names=["portQB_1","portQB_2","portCB_1","portCB_2","portCB_3"])
        node_C = Node("C",ID=2,port_names=["portQC_1","portCC_1"])



        # Quantum fibre
        MyQfiberAB=QuantumFibre("QFibre_A->B", length=fibre_len
            ,quantum_loss_model=None,p_loss_init=0, p_loss_length=0) 
        MyQfiberBC=QuantumFibre("QFibre_B->C", length=fibre_len
            ,quantum_loss_model=None,p_loss_init=0, p_loss_length=0) 
        # default value:
        # p_loss_init=0.2, p_loss_length=0.25,depolar_rate=0, c=200000, models=None


        # Classical fibre
        MyCfiberCB=DirectConnection("CFibreConn_C->B",
            ClassicalFibre("CFibre_C->B", length=fibre_len))
        MyCfiberBA=DirectConnection("CFibreConn_B->A",
            ClassicalFibre("CFibre_B->A", length=fibre_len))
        MyCfiberAB=DirectConnection("CFibreConn_A->B",
            ClassicalFibre("CFibre_A->B", length=fibre_len))

        # Quantum connection
        node_A.connect_to(node_B, MyQfiberAB,
            local_port_name =node_A.ports["portQA_1"].name,
            remote_port_name=node_B.ports["portQB_1"].name)
        node_B.connect_to(node_C, MyQfiberBC,
            local_port_name =node_B.ports["portQB_2"].name,
            remote_port_name=node_C.ports["portQC_1"].name)

        # Classical connection
        node_C.connect_to(node_B, MyCfiberCB,
            local_port_name="portCC_1", remote_port_name="portCB_3")
        node_B.connect_to(node_A, MyCfiberBA,
            local_port_name="portCB_1", remote_port_name="portCA_1")
        node_A.connect_to(node_B, MyCfiberAB,
            local_port_name="portCA_2", remote_port_name="portCB_2")



        myQLine=QLine(node_A=node_A,processor_A=processorA
            ,node_B=node_B,processor_B=processorB
            ,node_C=node_C,processor_C=processorC
            ,portNameQA1="portQA_1"
            ,portNameCA1="portCA_1",portNameCA2="portCA_2"

            ,portNameQB1="portQB_1",portNameQB2="portQB_2"
            ,portNameCB1="portCB_1",portNameCB2="portCB_2",portNameCB3="portCB_3"

            ,portNameQC1="portQC_1",portNameCC1="portCC_1")




        myQLine.start()

        ns.sim_run()
        if myQLine.keyA:
            QlineKeyListA.append(myQLine.keyA[0])
            QlineKeyListB.append(myQLine.keyB[0])
        
    return QlineKeyListA, QlineKeyListB


# test
run_QLine_sim(times=15)


    

