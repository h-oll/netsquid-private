#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
subQLine simulation 

Qubit sending direction
(A)------>(B)------>(C)
This sub-portocol is made for QLine. 

Function:
Generate classical key between A and B. 
'''


# In[9]:


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


# In[35]:


# get output value from a Qprogram
def getPGoutput(Pg,key):
    resList=[]
    tempDict=Pg.output
    value = tempDict.get(key, "")[0]
    return value


# A Quantum processor class which is able to send qubits from Qmemory
class sendableQProcessor(QuantumProcessor):
    def sendfromMem(self,inx,senderNode,receiveNode):
        payload=self.pop(inx)
        senderNode.get_conn_port(receiveNode.ID,label='Q').tx_output(payload)


# Quantum program operations on Node A 
class QG_A(QuantumProgram):
    def __init__(self):
        super().__init__()        
    def program(self):
        idx=self.get_qubit_indices(1)
        self.apply(INSTR_INIT, idx[0])
        yield self.run(parallel=False)


# In[44]:


class fQLine(Protocol):
    
    def __init__(self,nodeList,processorList):
        self.nodeList=nodeList
        self.processorList=processorList
        
        self.firstNode=nodeList[0]
        self.lastNode =nodeList[-1]
        self.firstProcessor=processorList[0]
        self.lastProcessor =processorList[-1]
        self.num_node=len(nodeList)
        self.fibre_len=1
        # create quantum fibre and connect
        
        
        QfibreList=[QuantumFibre(name="QF"+str(i),length=self.fibre_len) 
            for i in range(self.num_node-1)]
        for i in range(self.num_node-1):
            nodeList[i].connect_to(nodeList[i+1],QfibreList[i],label='Q')
    
    
        # create classical fibre and connect 
        CfibreList=[DirectConnection("CF"+str(i)
                    ,ClassicalFibre(name="CFibre_forth", length=self.fibre_len)
                    ,ClassicalFibre(name="CFibre_back" , length=self.fibre_len))
                        for i in range(self.num_node-1)]
        for i in range(self.num_node-1):
            nodeList[i].connect_to(nodeList[i+1],CfibreList[i],label='C')
        
        
        super().__init__()
        
        
    def start(self):
        super().start()
        
        self.Qubit_prepare(0,2)


# portocol operation ================================================================== 

    def Qubit_prepare(self,initNodeID,targetNodeID):
        print("Qubit_prepare")
        
        self.nodeList[targetNodeID].get_conn_port(
            self.nodeList[targetNodeID-1].ID,label='Q').bind_input_handler(
            self.QubitReceivePrepare)
        
        myQG_A=QG_A()
        self.processorList[initNodeID].execute_program(myQG_A,qubit_mapping=[0])
        self.processorList[initNodeID].set_program_done_callback(self.QubitSend,once=True)
        self.processorList[initNodeID].set_program_fail_callback(self.QG_Failed,once=True)
    
    
    def QG_Failed(self):
        print("QG_Failed")
        
        
    def QubitSend(self):
        print("QubitSend")
        self.processorList[0].sendfromMem(senderNode=self.nodeList[0]
            ,inx=[0]
            ,receiveNode=self.nodeList[1]) # get even index in qList
    
    def QubitReceivePrepare(self,qubit):
        print("received:",qubit.items)
        self.processorList[2].put(qubits=qubit.items)
        
        
    def ForwardSetting(self,initNodeID,targetNodeID,label=''):
        self.nodeList[initNodeID].get_conn_port(
            self.nodeList[initNodeID-1].ID,label=label).forward_input(
            self.nodeList[targetNodeID].get_conn_port(
            self.nodeList[initNodeID].ID,label=label))
        


# In[47]:


def run_QLine_sim(times=1,fibre_len=10**-3,noise_model=None): # fibre 1 m long
    
    
    # A B Hardware configuration
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
    
    
    node_A = Node("A",ID=0)
    node_B = Node("B",ID=1)
    node_C = Node("C",ID=2)
    
    for i in range(times):
        ns.sim_reset()
        
        myQLine=fQLine(nodeList=[node_A,node_B,node_C]
            ,processorList=[processorA,processorB,processorC])
        
        myQLine.ForwardSetting(1,2,'Q')
        
        myQLine.start()
        ns.sim_run()
        
    
    
#test
run_QLine_sim(times=1,fibre_len=10**-3,noise_model=None) 



# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:


def C_rec(mes):
    print("C received!")

    
ns.sim_reset()


        
fibre_len=1
node_A = Node("A",ID=0)
node_B = Node("B",ID=1)
node_C = Node("C",ID=2)

# Quantum fibre
MyQfiberAB=QuantumFibre("QFibre_A->B", length=fibre_len
    ,p_loss_init=0.04, p_loss_length=0.25)
MyQfiberBC=QuantumFibre("QFibre_B->C", length=fibre_len
    ,p_loss_init=0.04, p_loss_length=0.25)

# Quantum connection
node_A.connect_to(node_B, MyQfiberAB)
node_B.connect_to(node_C, MyQfiberBC) 
'''
,port_names=["portQAB","portCAB_1"]
,port_names=["portQBA","portQBC","portCBA_1"]
,port_names=["portQCB","portCCB_1"]
local_port_name =node_A.ports["portQAB"].name,
remote_port_name=node_B.ports["portQBA"].name
'''

#local_port_name declair makes get_conn_port fail

portAB=node_A.get_conn_port(node_B.ID)
portBA=node_B.get_conn_port(node_A.ID)
portBC=node_B.get_conn_port(node_C.ID)
portCB=node_C.get_conn_port(node_B.ID)

print("AB :",portAB.name) # only works on default ports
print("BA :",portBA.name)
print("BC :",portBC.name)
print("CB :",portCB.name)

'''
print("BA :",node_B.get_conn_port(node_A.ID))

'''
node_C.ports[portCB.name].bind_input_handler(C_rec)


node_A.ports[portAB.name].tx_output(create_qubits(1))

portBA.forward_input(portCB)
#node_B.ports[portBC.name].tx_output(create_qubits(1))



#print(type(node_A.get_conn_port(node_B.ID)))
ns.logger.setLevel(1)
ns.sim_run()

