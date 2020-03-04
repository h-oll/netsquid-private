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


# In[1]:


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


# In[2]:


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



# Quantum program operations usually on Node A 
class QG_I(QuantumProgram):
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



# Quantum program operations usually on Node B
class QG_T(QuantumProgram):
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
        


# In[17]:


class fQLine(Protocol):
    
    def __init__(self,nodeList,processorList,initNodeID,targetNodeID):
        self.nodeList=nodeList
        self.processorList=processorList
        
        self.initNodeID=initNodeID
        self.targetNodeID=targetNodeID
        
        self.num_node=len(nodeList)
        self.fibre_len=1
        
        # A rand
        self.r=randint(0,1)
        self.b=randint(0,1)
        # B rand
        self.c=randint(0,1)
        self.s=randint(0,1)
        
        self.keyAB=None
        self.keyBA=None
        
        self.TimeF=0.0
        self.TimeD=0.0
        
        # if not connected
        if self.nodeList[self.initNodeID].get_conn_port(
            self.targetNodeID,label='Q')== None:
        
            # create quantum fibre and connect
            QfibreList=[QuantumFibre(name="QF"+str(i),length=self.fibre_len) 
                for i in range(self.num_node-1)]
            for i in range(self.num_node-1):
                self.nodeList[i].connect_to(self.nodeList[i+1],QfibreList[i],label='Q')

            # create classical fibre and connect 
            CfibreList=[DirectConnection("CF"+str(i)
                        ,ClassicalFibre(name="CFibre_forth", length=self.fibre_len)
                        ,ClassicalFibre(name="CFibre_back" , length=self.fibre_len))
                            for i in range(self.num_node-1)]
            for i in range(self.num_node-1):
                nodeList[i].connect_to(nodeList[i+1],CfibreList[i],label='C')

        
        # forward setting
        #self.ForwardSetting(initNodeID,targetNodeID)
        #self.targetNodeID=initNodeID+1
        
        
        super().__init__()
        
        
    def start(self):
        super().start()
        
        # first callback
        self.nodeList[self.targetNodeID].get_conn_port(
            self.initNodeID,label='Q').bind_input_handler(
            self.T_QubitReceivePrepare)
        
        self.I_Qubit_prepare()


# portocol operation 1 =========================================== 

    def I_Qubit_prepare(self):
        #print("Qubit_prepare")
        
        myQG_I=QG_I(self.r,self.b)
        self.processorList[self.initNodeID].execute_program(
            myQG_I,qubit_mapping=[0])
        self.processorList[self.initNodeID].set_program_done_callback(
            self.I_QubitSend,once=True)
        self.processorList[self.initNodeID].set_program_fail_callback(
            self.I_QG_Failed,once=True)
    
    
    def I_QG_Failed(self):
        print("QG_Failed")
        
        
    def I_QubitSend(self):
        #print("QubitSend")
        self.nodeList[self.initNodeID].get_conn_port(
            self.initNodeID+1, label='C').bind_input_handler(self.I_Compare)
        
        
        self.processorList[self.initNodeID].sendfromMem(senderNode=self.nodeList[0]
            ,inx=[0]
            ,receiveNode=self.nodeList[1]) # get even index in qList

        
# 2  ==================================================

    def T_QubitReceivePrepare(self,qubit):
        #print("T received:",qubit.items)
        self.processorList[self.targetNodeID].put(qubits=qubit.items)
        self.myQG_T=QG_T(self.c,self.s)
        self.processorList[self.targetNodeID].execute_program(
            self.myQG_T,qubit_mapping=[0])
        self.processorList[self.targetNodeID].set_program_done_callback(
            self.T_SendsR,once=True)
        self.processorList[self.targetNodeID].set_program_fail_callback(
            self.T_PG_Fail,once=True)

        
    def T_PG_Fail(self):
        self.qubitLoss=True
        print("PG_targetFail!")
        


    def T_SendsR(self):
        #print("TargetSend s R")
        self.nodeList[self.targetNodeID].get_conn_port(
            self.initNodeID, label='C').bind_input_handler(self.T_Compare)
        
        self.R=getPGoutput(self.myQG_T,'R')
        self.nodeList[self.targetNodeID].get_conn_port(
            self.initNodeID,label='C').tx_output([self.s,self.R])
        


            
# 3 ==================================================        
        
    def I_Compare(self,alist): # first in the list is s, second is R
        #print("InitNodeCompare")
        if alist.items[0]==self.r: # if s == r
            self.keyAB=self.b
        self.I_Response()
        

    def I_Response(self):
        #print("InitNodeResponse")
        self.nodeList[self.initNodeID].get_conn_port(
            self.targetNodeID,label='C').tx_output(self.r)
      
    
# 4 =====================================================================

    def T_Compare(self,r):
        #print("TargetNodeCompare")
        self.qubitLoss=False
        if self.s==r.items[0]:
            self.keyBA=self.R ^ self.c
        self.TimeD=ns.util.simtools.sim_time(magnitude=ns.NANOSECOND)-self.TimeF 
        #in nanoseconds    
    
    
      

    def ForwardSetting(self,initNodeID,targetNodeID):
        #print("ForwardSetting")
        #port1.forward_input(port2)
        
        if initNodeID+2==targetNodeID:
            # forward forth
            self.nodeList[initNodeID+1].get_conn_port(
                initNodeID,label='Q').forward_input(
                self.nodeList[targetNodeID].get_conn_port(
                targetNodeID-1,label='Q'))
            self.nodeList[initNodeID+1].get_conn_port(
                initNodeID,label='C').forward_input(
                self.nodeList[targetNodeID].get_conn_port(
                targetNodeID-1,label='C'))
            # forward back
            self.nodeList[targetNodeID-1].get_conn_port(
                targetNodeID,label='C').forward_input(
                self.nodeList[initNodeID].get_conn_port(
                targetNodeID-1,label='C'))
        else:
            self.ForwardSetting(initNodeID+1,targetNodeID)
        
        
        
        
        
        
        


# In[18]:


def run_QLine_sim(times=1,fibre_len=10**-3,noise_model=None): # fibre 1 m long
    
    
    keyListAB=[]
    keyListBA=[]
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
        # A B Hardware configuration
        myQLine=fQLine(nodeList=[node_A,node_B,node_C]
            ,processorList=[processorA,processorB,processorC]
            ,initNodeID=0,targetNodeID=1)
        #myQLine.ForwardSetting(0,2)
        myQLine.start()
        ns.sim_run()
        if myQLine.keyAB !=None :
            keyListAB.append(myQLine.keyAB)
            keyListBA.append(myQLine.keyBA)
        
    print(keyListAB)
    print(keyListBA)
    
    '''
    for i in range(times):
        ns.sim_reset()
        
        myQLine=fQLine(nodeList=[node_A,node_B,node_C]
            ,processorList=[processorA,processorB,processorC])
        myQLine.ForwardSetting(1,2)
        
        myQLine.start()
        ns.sim_run()
    '''
        
#test
run_QLine_sim(times=20,fibre_len=10**-3,noise_model=None) 



# In[ ]:





# In[ ]:





# In[ ]:




