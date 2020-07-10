#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.protocols import NodeProtocol
from netsquid.qubits.operators import X,H,Z
from netsquid.nodes.connections import DirectConnection
from netsquid.components  import ClassicalFibre,QuantumFibre
from netsquid.components.models import  FibreDelayModel
from netsquid.qubits.qformalism import *
from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.components.models.qerrormodels import *

from netsquid.components import QSource,Clock
from netsquid.components.qsource import SourceStatus

from random import randint


# In[2]:


# processor class
class sendableQProcessor(QuantumProcessor):
    def sendfromMem(self,inx,senderNode,senderPortName):
        payload=self.pop(inx)
        senderNode.ports[senderPortName].tx_output(payload)


# In[3]:


# client protocol
class ClientProtocol(NodeProtocol):
    
    def __init__(self,node,processor,raw,column
            ,port_names=["portQA_1","portCA_1","portCA_2"]):
        super().__init__()
        self.node=node
        self.processor=processor
        self.raw=raw
        self.column=column
        self.qList=[]
        self.portNameQ1=port_names[0]
        self.qubitCount=0
        
    def sendServer(self):
        print("in sendServer")
        self.processor.sendfromMem(senderNode=self.node
            ,inx=list(range(0,self.raw*self.column,1))   # send odd index in qList
            ,senderPortName=self.node.ports[self.portNameQ1].name)

    # move qubits from qsource to qmem in qprocessor
    def storeSourceOutput(self,qubit):
        self.qList.append(qubit.items[0])
        
        if len(self.qList)==self.raw*self.column:
            print("putting qList to",self.qList)
            self.processor.put(qubits=self.qList)
            
            self.sendServer()
        
        
    def run(self):
        print("client on")
        clientSource = QSource("client_source"
            ,status=SourceStatus.EXTERNAL) # enable frequency
        clientSource.ports["qout0"].bind_output_handler(
            self.storeSourceOutput)
        
        #set clock
        clock = Clock("clock", frequency=1e9, max_ticks=self.raw*self.column)
        clock.ports["cout"].connect(clientSource.ports["trigger"])
        clock.start()
        


# In[4]:


# server protocol
class ServerProtocol(NodeProtocol):
    
    def __init__(self,node,processor,raw,column):
        super().__init__()
        self.node=node
        self.processor=processor
        self.raw=raw
        self.column=column

    def run(self):
        print("server on")
        port = self.node.ports["portQB_1"]
        qubitList=[]
        
        #receive qubits from client
        yield self.await_port_input(port)
        qubitList.append(port.rx_input().items) 
        print("B received qubits:",qubitList)


# In[5]:


# implement
def run_UBQC_sim(runtimes=1,num_bits=20,fibre_len=10**-9,noise_model=None,
               loss_init=0,loss_len=0):
    
    ns.sim_reset()
    # nodes====================================================================

    nodeA = Node("Alice", port_names=["portQA_1","portCA_1","portCA_2"])
    nodeB = Node("Bob"  , port_names=["portQB_1","portCB_1","portCB_2"])

    # processors===============================================================
    
    Alice_processor=sendableQProcessor("processor_A", num_positions=5,
        mem_noise_models=None, phys_instructions=[
        PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
        PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
        PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=noise_model),
        PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
        PhysicalInstruction(INSTR_CNOT,duration=1,q_noise_model=noise_model),
        PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True),
        PhysicalInstruction(INSTR_MEASURE_X, duration=1, parallel=True)])


    Bob_processor=sendableQProcessor("processor_B", num_positions=100,
        mem_noise_models=None, phys_instructions=[
        PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
        PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
        PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=noise_model),
        PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
        PhysicalInstruction(INSTR_CNOT,duration=1,q_noise_model=noise_model),
        PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True),
        PhysicalInstruction(INSTR_MEASURE_X, duration=1, parallel=True)])

    
    # fibres==================================================================
        
        
    MyQfiber=QuantumFibre("QFibre_A->B", length=fibre_len,
        quantum_loss_model=None,p_loss_init=loss_init, p_loss_length=loss_len) 
    nodeA.connect_to(nodeB, MyQfiber,
        local_port_name =nodeA.ports["portQA_1"].name,
        remote_port_name=nodeB.ports["portQB_1"].name)


    MyCfiber =DirectConnection("CFibreConn_B->A",
                ClassicalFibre("CFibre_B->A", length=fibre_len))
    MyCfiber2=DirectConnection("CFibreConn_A->B",
                ClassicalFibre("CFibre_A->B", length=fibre_len))

    nodeB.connect_to(nodeA, MyCfiber,
                        local_port_name="portCB_1", remote_port_name="portCA_1")
    nodeA.connect_to(nodeB, MyCfiber2,
                        local_port_name="portCA_2", remote_port_name="portCB_2")

    
    
    
    Alice=ClientProtocol(nodeA,Alice_processor,2,2)
    BOb=ServerProtocol(nodeB,Bob_processor,2,2)
    Alice.start()
    BOb.start()
    #ns.logger.setLevel(1)
    ns.sim_run()
    
    #print(Alice.qList)
    
run_UBQC_sim()


# In[6]:


# plot 


# In[ ]:




