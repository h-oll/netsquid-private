#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
QLine simulation 

Qubit sending direction
(A)------>(B)------>(C)------>(D)

According to roles in Qline, we have
(O)------>(I)------>(T)------>(D)
Indicating Origin,Initial,Target and Destiny, where (I) can be (O) 
as well as (T) being (D).

Origin : First Node of this QLine.
Initial: The initial node that start to generate a key pair. 
Target : Target node that share a key pair with the initial node. 
Destiny: Last node of this QLine.



Objective:
Establishes a shared key between A and B. 
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


# In[12]:



def CheckConnection(NodeList):
    for node in NodeList:
        for otherNode in NodeList:
            if node.get_conn_port(otherNode.ID,label='Q') :
                print(node.get_conn_port(otherNode.ID,label='Q'))
            if node.get_conn_port(otherNode.ID,label='C'):
                print(node.get_conn_port(otherNode.ID,label='C'))
    print("==================")

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
class QG_O(QuantumProgram):
    def __init__(self):
        super().__init__()
        
    def program(self):
        idx=self.get_qubit_indices(1)
        self.apply(INSTR_INIT, idx[0])
        yield self.run(parallel=False)

        
# Quantum program in initNode!=0 case
class QG_I(QuantumProgram):
    def __init__(self,r,b):
        self.r=r
        self.b=b
        super().__init__()
        
    def program(self):
        idx=self.get_qubit_indices(1)
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
        
        yield self.run(parallel=False)
        
        
class QG_D(QuantumProgram):
    def __init__(self):
        super().__init__()

    def program(self):
        idx=self.get_qubit_indices(1)
        self.apply(INSTR_MEASURE, qubit_indices=idx[0], output_key='R',physical=True) 
        
        yield self.run(parallel=False)


# In[13]:


class fQLine(Protocol):
    def __init__(self,nodeList,processorList,initNodeID,targetNodeID):
        self.nodeList=nodeList
        self.processorList=processorList
        
        self.initNodeID=initNodeID
        self.targetNodeID=targetNodeID
        
        self.num_node=len(nodeList)
        self.fibre_len=0
        
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
        
        # if not forwarded, connect them with Q,C fibres
        if self.nodeList[self.initNodeID].get_conn_port(
            self.initNodeID+1,label='Q')== None:
            
            # create quantum fibre and connect
            self.QfibreList=[QuantumFibre(name="QF"+str(i),length=self.fibre_len,
                p_loss_init=0,p_loss_length=0) 
                for i in range(self.num_node-1)]
            
            for i in range(self.num_node-1):
                self.nodeList[i].connect_to(
                    self.nodeList[i+1],self.QfibreList[i],label='Q')
            
            
            # create classical fibre and connect 
            CfibreList=[DirectConnection("CF"+str(i)
                        ,ClassicalFibre(name="CFibre_forth", length=self.fibre_len)
                        ,ClassicalFibre(name="CFibre_back" , length=self.fibre_len))
                            for i in range(self.num_node-1)]
            for i in range(self.num_node-1):
                self.nodeList[i].connect_to(
                    self.nodeList[i+1],CfibreList[i],label='C')
        
        super().__init__()




    '''
    Fuction used to modify ports directions.
    To make indirect path for nodes.
    The two nodes could have multiple nodes in between.
    input:
        initNodeID:  The node which initiate this QKD.
        targetNodeID: The target node we want to have shared key with. 
    '''
    def ForwardSetting(self,initNodeID,targetNodeID):
        #print("ForwardSetting")
        if not self.nodeList[initNodeID+1].get_conn_port(
            initNodeID,label='Q').forwarded_ports: 
            # if not done before(for multiple bits)
            for i in range(self.initNodeID,targetNodeID-1):
                self.nodeList[i+1].get_conn_port(
                    i,label='Q').forward_input(self.QfibreList[i+1].ports["send"]) 
                self.nodeList[i+1].get_conn_port(
                    i,label='C').forward_input(
                    self.nodeList[i+2].get_conn_port(
                    i+1,label='C'))
    
            for i in range(self.targetNodeID-2,self.initNodeID-1,-1):
                self.nodeList[i+1].get_conn_port(
                    i+2,label='C').forward_input(
                    self.nodeList[i].get_conn_port(
                    i+1,label='C'))


    def start(self):
        super().start()
        
        # forware setting based on position of two nodes
        if self.initNodeID+1!=self.targetNodeID:
            self.ForwardSetting(self.initNodeID,self.targetNodeID)
        if self.initNodeID>=1:
            self.ForwardSetting(0,self.initNodeID)
        if self.targetNodeID<len(self.nodeList)-2:
            self.ForwardSetting(
                self.targetNodeID,self.nodeList[len(self.nodeList)].ID)
        
        # T callback
        self.nodeList[self.targetNodeID].get_conn_port(
            self.targetNodeID-1,label='Q').bind_input_handler(
            self.T_Rec_HX)
        
        self.O_Qubit_prepare()
        

# O =========================================== 

    def O_Qubit_prepare(self):
        #print("O_Qubit_prepare")
        myQG_O=QG_O()
        self.processorList[0].execute_program(
            myQG_O,qubit_mapping=[0])
        self.processorList[0].set_program_done_callback(
            self.O_QubitSend,once=True)
        self.processorList[0].set_program_fail_callback(
            self.O_QG_Failed,once=True)
        
        
        if self.initNodeID==0: # if I==O !!!
            self.processorList[0].set_program_done_callback(
                self.I_QG_HX,once=True)
        else:
            self.processorList[0].set_program_done_callback(
                self.O_QubitSend,once=True)
        
    
    def O_QG_Failed(self):
        print("QG_Failed")
    
    
    # when O != I
    def O_QubitSend(self):
        #print("O_QubitSend")
        # I callback
        self.nodeList[self.initNodeID].get_conn_port(
            self.initNodeID-1,label='Q').bind_input_handler(
            self.I_rec)
        # send qubits
        self.processorList[0].sendfromMem(
            senderNode=self.nodeList[0],inx=[0]
            ,receiveNode=self.nodeList[1]) 

# I  ==================================================
    def I_rec(self,qubit):
        #print("I_rec")
        self.processorList[self.initNodeID].put(qubits=qubit.items)
        self.I_QG_HX()
        
    def I_QG_HX(self):
        #print("I_QG_HX")
        myQG_I=QG_I(self.r,self.b)
        self.processorList[self.initNodeID].execute_program(
            myQG_I,qubit_mapping=[0])
        self.processorList[self.initNodeID].set_program_done_callback(
            self.I_QubitSend,once=True)
        self.processorList[self.initNodeID].set_program_fail_callback(
            self.I_QG_Failed,once=True)
        
    def I_QG_Failed(self):
        print("QG_I_Failed")     
        
    def I_QubitSend(self):
        #print("I_QubitSend")
        # I callback
        self.nodeList[self.initNodeID].get_conn_port(
            self.initNodeID+1, label='C').bind_input_handler(self.I_Compare)

        self.processorList[self.initNodeID].sendfromMem(
            senderNode=self.nodeList[self.initNodeID],inx=[0]
            ,receiveNode=self.nodeList[self.initNodeID+1]) # get even index in qList


# T  ==================================================
    def T_PG_Fail(self):
        self.qubitLoss=True
        print("PG_targetFail!")
        
    def T_Rec_HX(self,qubit):
        #print("T_Rec_HX: ",qubit.items)
        self.processorList[self.targetNodeID].put(qubits=qubit.items)
        self.myQG_T=QG_T(self.c,self.s)
        
        # T HX
        self.processorList[self.targetNodeID].execute_program(
            self.myQG_T,qubit_mapping=[0])
        self.processorList[self.targetNodeID].set_program_fail_callback(
            self.T_PG_Fail,once=True)
        
        # if T==D !!!
        if self.targetNodeID==len(self.nodeList)-1: 
            # send s R back to T then I
            #len(self.nodeList)-1==self.targetNodeID
            self.processorList[self.targetNodeID].set_program_done_callback(
                self.D_meas,once=True)
        
        else: # T!=D !!! send forward
            # set T callback for R
            self.nodeList[self.targetNodeID].get_conn_port(
                self.targetNodeID+1, label='C').bind_input_handler(self.T_recR)
            # set callback for D
            self.nodeList[-1].get_conn_port(
                len(self.nodeList)-2, label='Q').bind_input_handler(self.D_rec)
            
            #send to D
            self.processorList[self.targetNodeID].set_program_done_callback(
                self.T_Send_D,once=True)
            # get even index in qList


        
    # T!=D
    def T_Send_D(self):
        #print("T_Send_D")
        self.processorList[self.targetNodeID].sendfromMem(
            senderNode=self.nodeList[self.targetNodeID],inx=[0]
            ,receiveNode=self.nodeList[self.targetNodeID+1])
        #self.nodeList[self.targetNodeID].get_conn_port(
        #    self.targetNodeID-1, label='C').bind_input_handler(self.T_Compare)

#  D ==================================================
    # must
    def D_meas(self):
        #print("D_meas")
        self.myQG_D=QG_D()
        # lastnode.ID==len(self.nodeList)-1
        #print(self.nodeList[-1])
        #print(self.nodeList[len(self.nodeList)-1])
        self.processorList[-1].execute_program(
            self.myQG_D,qubit_mapping=[0])
        self.processorList[-1].set_program_fail_callback(
            self.D_Fail,once=True)
        self.processorList[-1].set_program_done_callback(
            self.D_sendback,once=True)

    def D_Fail(self):
        print("D_Fail")

    # if T!=D
    def D_rec(self,qubit):
        #print("D_rec")
        self.processorList[-1].put(qubits=qubit.items)
        self.D_meas()

    # must
    def D_sendback(self):
        #print("D_sendback")
        self.R=getPGoutput(self.myQG_D,'R')
        self.nodeList[len(self.nodeList)-1].get_conn_port(
            len(self.nodeList)-2,label='C').tx_output([self.s,self.R])
        
# C1 ==================================================

    def T_recR(self,R):
        # receive R then send back
        self.nodeList[self.targetNodeID].get_conn_port(
            self.targetNodeID-1,label='C').tx_output([self.s,R.items[0]])
        
    
# C2 ==================================================    
    def I_Compare(self,alist): # receives [self.s,self.R]
        #print("InitNodeCompare")
        if alist.items[0]==self.r: # if s == r
            self.keyAB=self.b
        self.I_Response()

    def I_Response(self):
        #print("InitNodeResponse")
        # callback T compare
        self.nodeList[self.targetNodeID].get_conn_port(
            self.targetNodeID-1, label='C').bind_input_handler(self.T_Compare)

        self.nodeList[self.initNodeID].get_conn_port(
            self.initNodeID+1,label='C').tx_output(self.r)
      
    
# C3 =====================================================================

    def T_Compare(self,r):
        #print("TargetNodeCompare")
        self.qubitLoss=False
        if self.s==r.items[0]:
            self.keyBA=self.R ^ self.c
        self.TimeD=ns.util.simtools.sim_time(magnitude=ns.NANOSECOND)-self.TimeF 
        #in nanoseconds    


# In[16]:



def run_QLine_sim(times=1,fibre_len=10**-3,noise_model=None): # fibre 1 m long
    
    keyListAB=[]
    keyListBA=[]
    
    # A B Hardware configuration
    processorA=sendableQProcessor("processor_A", num_positions=1,
                mem_noise_models=None, phys_instructions=[
                PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
                PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
                PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
                PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True)])

    #mem_noise_models=[DepolarNoiseModel(0)] * 100
    processorB=sendableQProcessor("processor_B", num_positions=1,
                mem_noise_models=None, phys_instructions=[
                PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
                PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
                PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
                PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True)])

    processorC=sendableQProcessor("processor_C", num_positions=1,
                mem_noise_models=None, phys_instructions=[
                PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
                PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
                PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
                PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True)])

    processorD=sendableQProcessor("processor_D", num_positions=1,
                mem_noise_models=None, phys_instructions=[
                PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
                PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
                PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
                PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True)])

    node_A = Node("A",ID=0)
    node_B = Node("B",ID=1)
    node_C = Node("C",ID=2)
    node_D = Node("D",ID=3)
    
    for i in range(times):
        ns.sim_reset()
        myQLine=fQLine(nodeList=[node_A,node_B,node_C,node_D]
            ,processorList=[processorA,processorB,processorC,processorD]
            ,initNodeID=1,targetNodeID=3)
        #myQLine.ForwardSetting(0,3)
        myQLine.start()
        ns.sim_run()
        if myQLine.keyAB != None :
            keyListAB.append(myQLine.keyAB)
            keyListBA.append(myQLine.keyBA)
    print(keyListAB)
    print(keyListBA)


#test
#ns.logger.setLevel(1)
run_QLine_sim(times=10,fibre_len=10**-3,noise_model=None) 


# In[ ]:





# In[ ]:




