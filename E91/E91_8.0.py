#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.protocols import Protocol,LocalProtocol,NodeProtocol
from netsquid.qubits import create_qubits
from netsquid.qubits.operators import X,H,Z
from netsquid.nodes.connections import DirectConnection
from netsquid.components  import ClassicalFibre,QuantumFibre,QuantumMemory
from netsquid.components.models import  FibreDelayModel
from netsquid.pydynaa import Entity,EventHandler,EventType
from netsquid.components.qchannel import QuantumChannel

from random import randint

from netsquid.qubits.qformalism import *
from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.components.models.qerrormodels import *


# In[2]:


# General function


'''
Simply returns a list with 0 or 1 in given length.
'''
def Random_basis_gen(length):
    return [randint(0,1) for i in range(length)]



'''
Compare two lists, find the unmatched index, 
    then remove corresponding slots in loc_meas.
Input:
    loc_basis_list: local basis used for measuring qubits.(list of int)
    rem_basis_list: remote basis used for measuring qubits.(list of int)
        Two lists with elements 0-2 (Z,X, -1:qubit missing).
        Two lists to compare.
        
    loc_meas: Local measurement results to keep.(list of int)
Output:
    measurement result left.
'''

def Compare_basis(loc_basis_list,rem_basis_list,loc_res):

    if len(loc_basis_list) != len(rem_basis_list): #should be  len(num_bits)
        print("Comparing error! length of basis does not match! Check classical port")
        return -1
    
    popList=[]
    
    for i in range(len(rem_basis_list)):
        if loc_basis_list[i] != rem_basis_list[i]:
            popList.append(i)
    
    for i in reversed(popList): 
        if loc_res:
            loc_res.pop(i)
        
    return loc_res



'''
input:
    Pg: A quantum program (QuantumProgram)
output:
    resList: A list of outputs from the given quantum program, also sorted by key.(list of int)
'''

def getPGoutput(Pg):
    resList=[]
    tempDict=Pg.output
    if "last" in tempDict:
        del tempDict["last"]
        
    # sort base on key
    newDict=sorted({int(k) : v for k, v in tempDict.items()}.items())
    
    #take value
    for k, v in newDict:
        resList.append(v[0])
    
    return resList


'''
To add value -1 on tarList at positions given by lossList.
input:
    lossList: A list indecates positions to add -1. (list of int)
    tarList: List to add -1. (list of any)
output:
    tarList: target List.(list of any)
'''
def AddLossCase(lossList,tarList):
    for i in range(len(lossList)):
        tarList.insert(lossList[i],-1)
    return tarList



# In[3]:


# processor class
class sendableQProcessor(QuantumProcessor):
    def sendfromMem(self,inx,senderNode,senderPortName):
        payload=self.pop(inx)
        senderNode.ports[senderPortName].tx_output(payload)



# class of quantum program
class QG_A_qGen(QuantumProgram):
    
    def __init__(self,num_bits=1):
        self.num_bits=num_bits
        super().__init__()
        
    def program(self):
        qList_idx=self.get_qubit_indices(2*self.num_bits)
        
        # create multiEPR
        for i in range(2*self.num_bits):
            self.apply(INSTR_INIT, qList_idx[i])
            if i%2==0:                           # List A case
                self.apply(INSTR_H, qList_idx[i])
            else:                                # List B case
                self.apply(INSTR_CNOT, [qList_idx[i-1], qList_idx[i]])
                
        yield self.run(parallel=False)


class QG_A_measure(QuantumProgram):
    def __init__(self,basisList,num_bits):
        self.basisList=basisList
        self.num_bits=num_bits
        super().__init__()


    def program(self):   
        for i in range(0,len(self.basisList*2),2):
            if self.basisList[int(i/2)] == 0:                  # standard basis
                self.apply(INSTR_MEASURE, 
                    qubit_indices=i, output_key=str(i),physical=True) 
            else:                              # 1 case # Hadamard basis
                self.apply(INSTR_MEASURE_X, 
                    qubit_indices=i, output_key=str(i),physical=True) 
        yield self.run(parallel=False)
        
        

class QG_B_measure(QuantumProgram):
    def __init__(self,basisList,num_bits):
        self.basisList=basisList
        self.num_bits=num_bits
        super().__init__()


    def program(self):   
        for i in range(len(self.basisList)):
            if self.basisList[i] == 0:                  # standard basis
                self.apply(INSTR_MEASURE, 
                    qubit_indices=i, output_key=str(i),physical=True) 
            else:                              # 1 case # Hadamard basis
                self.apply(INSTR_MEASURE_X, 
                    qubit_indices=i, output_key=str(i),physical=True) 
        yield self.run(parallel=False)


# In[4]:


class AliceProtocol(NodeProtocol):
    
    def A_sendEPR(self):
        self.processor.sendfromMem(senderNode=self.node
            ,inx=list(range(1,2*self.num_bits+1,2))   # send odd index in qList
            ,senderPortName=self.node.ports[self.portNameQ1].name)

    
    def A_getPGoutput(self):
        self.loc_mesRes=getPGoutput(self.myQG_A_measure)
        
        
        
        
    def __init__(self,node,processor,num_bits,
                port_names=["portQA_1","portCA_1","portCA_2"]):
        super().__init__()
        self.num_bits=num_bits
        self.node=node
        self.processor=processor
        self.portNameQ1=port_names[0]
        self.EPRList=None
        self.basisList=Random_basis_gen(self.num_bits)
        self.loc_mesRes=[]
        self.key=None
        
    # =======================================A run ============================
    def run(self):
        #print("Starting Alice at t={}".format(ns.sim_time()))
        
        # Do Qprogram
        PG_A=QG_A_qGen(num_bits=self.num_bits)
        self.processor.execute_program(
            PG_A,qubit_mapping=[i for  i in range(0, 2*self.num_bits)])
        
        # send half EPR to B
        self.processor.set_program_done_callback(self.A_sendEPR,once=True)
        # measured and save on back
        
        # receive B basis
        port=self.node.ports["portCA_1"]
        yield self.await_port_input(port)
        basis_B = port.rx_input().items
        
        
        
        #self.A_measure()
        self.myQG_A_measure=QG_A_measure(
            basisList=self.basisList,num_bits=self.num_bits)
        self.processor.execute_program(
            self.myQG_A_measure,qubit_mapping=[i for  i in range(0, 2*self.num_bits)])
        
        
        # get A meas
        self.processor.set_program_done_callback(
            self.A_getPGoutput,once=True)
        
        # send A basis to B
        self.node.ports["portCA_2"].tx_output(self.basisList)
        
        
        # compare basis
        
        yield self.await_program(processor=self.processor)
        self.loc_mesRes=Compare_basis(self.basisList,basis_B,self.loc_mesRes)
        
        
        self.key=''.join(map(str, self.loc_mesRes))
        print("A key:",self.key)

        


# In[5]:


class BobProtocol(NodeProtocol):
    
    def B_checkLoss(self,qList):
        num_inx=int(qList[0].name[3:-len('-')-1]) # get index from bits

        self.lossList=[]
        for idx,qubit in enumerate(qList):
            loc_num=int(qubit.name[3:-len('-')-1]) # received qubit
            found_flag=True
            while(found_flag and len(self.lossList)<self.num_bits):
                if loc_num==num_inx:
                    found_flag=False
                else:
                    self.lossList.append(idx)
                num_inx+=2
        # init B's basisList
        self.basisList=Random_basis_gen(len(qList))
        
        # check for first N qubit loss
        if self.num_bits-len(self.lossList)>len(qList):
            # first qubit loss detected
            # value of self.firstLoss indecats how many qubits are lost
            self.firstLoss=self.num_bits-len(qList)-len(self.lossList)  
        else:
            self.firstLoss=0

    
    def B_getPGoutput(self):
        self.loc_measRes=getPGoutput(self.myQG_B_measure)



    def __init__(self,node,processor,num_bits,
                port_names=["portQB_1","portCB_1","portCB_2"]):
        super().__init__()
        self.num_bits=num_bits
        self.node=node
        self.processor=processor
        self.qList=None
        #self.basisList=Random_basis_gen(self.num_bits)
        self.loc_measRes=[-1]*self.num_bits   # init value assume that all qubits are lost
        self.key=None
        self.PG_B=None
        self.lossList=[]
        self.firstLoss=0
        

    # =======================================B run ============================
    def run(self):
        port = self.node.ports["portQB_1"]
        qubitList=[]
        
        #receive qubits from A
        yield self.await_port_input(port)
        qubitList.append(port.rx_input().items) 
        #print("B received qubits:",qubitList)
        self.B_checkLoss(qubitList[0])
        
        
        #put qubits into B memory
        for qubit in qubitList:
            self.processor.put(qubit)
        
        
        # self.B_meas()
        self.myQG_B_measure=QG_B_measure(
            basisList=self.basisList,num_bits=self.num_bits)
        self.processor.execute_program(
            self.myQG_B_measure,qubit_mapping=[i for  i in range(0,self.num_bits)])
        
        # get meas result
        self.processor.set_program_done_callback(self.B_getPGoutput,once=True)
        #yield self.await_program(processor=self.processor)
        #self.loc_measRes=getPGoutput(self.myQG_B_measure)
        
        yield self.await_program(processor=self.processor)
        
        # add Loss case
        self.loc_measRes=AddLossCase(self.lossList,self.loc_measRes)
        self.basisList=AddLossCase(self.lossList,self.basisList)
        
        # self.B_send_basis()
        if self.firstLoss>=1:   
            for i in range(self.firstLoss):
                self.loc_measRes.insert(0,-1)
                self.basisList.insert(0,-1)
        
        
        self.node.ports["portCB_1"].tx_output(self.basisList)
        
        # wait for A's basisList
        port=self.node.ports["portCB_2"]
        yield self.await_port_input(port)
        basis_A=port.rx_input().items
        #print("B received basis_A:",basis_A)
        
        self.loc_measRes=Compare_basis(self.basisList,basis_A,self.loc_measRes)
        
        
        self.key=''.join(map(str, self.loc_measRes))
        print("B key:",self.key)
        
        


# In[6]:


ns.sim_reset()


# nodes====================================================================

nodeA = Node("Alice", port_names=["portQA_1","portCA_1","portCA_2"])
nodeB = Node("Bob"  , port_names=["portQB_1","portCB_1","portCB_2"])

# processors====================================================================
noise_model=None
Alice_processor=sendableQProcessor("processor_A", num_positions=100,
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


# fibres=======================================================================

fibre_len=1
MyQfiber=QuantumFibre("QFibre_A->B", length=fibre_len,quantum_loss_model=None,
    p_loss_init=0, p_loss_length=0) 
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




# In[7]:


num_bits=40
Alice_protocol = AliceProtocol(nodeA,Alice_processor,num_bits)
Bob_protocol = BobProtocol(nodeB,Bob_processor,num_bits)
Alice_protocol.start()
Bob_protocol.start()
#ns.logger.setLevel(1)
stats = ns.sim_run()


# In[ ]:





# In[ ]:




