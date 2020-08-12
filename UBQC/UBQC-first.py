#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.protocols import NodeProtocol
from netsquid.qubits.operators import X,H,Z
from netsquid.qubits.operators import create_rotation_op
from netsquid.nodes.connections import DirectConnection
from netsquid.components  import QuantumMemory,ClassicalFibre,QuantumFibre
from netsquid.components.models import  FibreDelayModel
from netsquid.qubits.qformalism import *
from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.components.models.qerrormodels import *

from netsquid.components import QSource,Clock
from netsquid.components.qsource import SourceStatus


from random import randint
import json


# In[67]:


# General function
def load_circuit(path):
    with open(path, "r") as circ:
        circuit = json.load(circ)
    nGates = len(circuit["gates"])
    gates = []
    qubits = []
    qubits_1 = []
    qubits_2 = []
    angles = []
    for g in range(0, nGates):
        qubits = qubits + circuit["gates"][g]["qbits"]
        qubits_1 = qubits_1 + [int(circuit["gates"][g]["qbits"][0])]
        if len(circuit["gates"][g]["qbits"]) == 1:
            qubits_2 = qubits_2 + [0]
        else:
            qubits_2 = qubits_2 + [int(circuit["gates"][g]["qbits"][1])]
        gates = gates + [circuit["gates"][g]["type"]]
        if gates[g] == 'T' :
            angles = angles + [32]
        if gates[g] == 'R_Z':
            angles = angles + [int(circuit["gates"][g]["angle"])]

    #print("qubits {}".format(qubits))
    nqbits = len(set(qubits))
    return gates, [qubits_1, qubits_2], nqbits, angles


# input: qList: a list of qubit. columns:columns, raws:raws
# output: a 2D array list of qubit entangled in brickwork.
def make2DqList(qList,columns,raws):
    res=[]
    for r in range(raws):
        aRaw=[]
        for c in range(columns):
            aRaw.append(qList.pop(0))
        res.append(aRaw)
    return res
    


# In[68]:


# processor class
class sendableQProcessor(QuantumProcessor):
    def sendfromMem(self,inx,senderNode,senderPortName):
        payload=self.pop(inx)
        senderNode.ports[senderPortName].tx_output(payload)

        
# apply Cz on qubits in BrickWork manner
class QG_ServerMakeBrickWork(QuantumProgram):
    
    def __init__(self,raw,column):
        self.raw=raw
        self.column=column
        super().__init__()

    def program(self):
        for r in range(self.raw):
            for c in range(self.column-1):
                # horizontal links
                self.apply(INSTR_CZ, 
                    qubit_indices=[r*self.column+c,r*self.column+c+1], physical=True) 
                    # r*self.column+c=1D index : convert 2D to 1D
                # vertical links
                if c%8 == 2 and r%2 == 1: # link up
                    self.apply(INSTR_CZ, 
                        qubit_indices=[(r-1)*self.column+c,r*self.column+c], physical=True)
                if c%8 == 6 and r%2 == 0 and r!=0: # link up
                    self.apply(INSTR_CZ, 
                        qubit_indices=[(r-1)*self.column+c,r*self.column+c], physical=True)
                    
        yield self.run(parallel=False)
        
        
        
class QG_ClientPrepareQubits(QuantumProgram):
    
    def __init__(self,raw,column):
        self.raw=raw
        self.column=column
        super().__init__()
        
    def program(self):
        for i in range(0,self.raw * self.column):
            self.apply(INSTR_H,qubit_indices=i,physical=True) 
        yield self.run(parallel=False)


# In[85]:


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
    # should not needed latter
    def recordSourceOutput(self,qubit):
        
        print("recording SourceOutput")
        self.qList.append(qubit.items[0])
        
        if len(self.qList)==self.raw*self.column:
            print("putting qList to",self.qList)
            self.processor.put(qubits=self.qList)
            #prepare +> state
            self.processor.execute_program(QG_ClientPrepareQubits(self.raw,self.column)
                ,qubit_mapping=[i for  i in range(self.raw*self.column)])
            
            
            
            


    def run(self):
        print("client on")
        path="circuits/circuit1.json"
        #self.prepareQubits(path)
        load_circuit(path)
        
        
        clientSource = QSource("client_source"
            ,status=SourceStatus.EXTERNAL) # enable frequency
        clientSource.ports["qout0"].bind_output_handler(
            self.recordSourceOutput)
        
        #set clock
        clock = Clock("clock", frequency=1e9, max_ticks=self.raw*self.column)
        clock.ports["cout"].connect(clientSource.ports["trigger"])
        clock.start()
        
        yield self.await_program(processor=self.processor)
        
        self.sendServer()


# In[86]:


# server protocol
class ServerProtocol(NodeProtocol):
    
    def __init__(self,node,processor,raw,column):
        super().__init__()
        self.node=node
        self.processor=processor
        self.raw=raw
        self.column=column
        self.brickwork=[]
        self.brickwork2D=[]
        self.serverMemoryList=[]
        for i in range(self.raw):
            self.serverMemoryList.append(QuantumMemory("serverMemory",self.column
                ,ID=i,memory_noise_models=None))  
            
            #column = length of Qmem here
            #DepolarNoiseModel(depolar_rate=depolar_rate,time_independent=timeIND)
        
    #def entangle
        
        
    def measureByAngle(self,raw,column,angle):
        #(self, positions=None, observable=Z, list meas_operators=None, 
        # bool discard=False, bool skip_noise=False, bool check_positions=True)
        
        tmpOP=create_rotation_op(angle, rotation_axis=(0, 0, 1), 
            conjugate=False, name=None, cacheable=True)
        
        return self.serverMemoryList[raw].measure(positions=column,observable=tmpOP)
    
    
    def run(self):
        print("server on")
        port = self.node.ports["portQB_1"]
        
        #receive qubits from client
        yield self.await_port_input(port)
        
        self.brickwork=port.rx_input().items
        #self.brickwork.append(port.rx_input().items)
        print("server received qubits:",self.brickwork)
        
        
        # make received qubits in 2D array for brickwork
        #self.brickwork2D=make2DqList(self.brickwork,self.column,self.raw)
        #print("server make 2D list of qubits:",self.brickwork2D)
        
        
        # server 
        for qubits in self.brickwork:
            self.processor.put(qubits=qubits)
        
        self.processor.execute_program(
            QG_ServerMakeBrickWork(self.raw,self.column)
                ,qubit_mapping=[i for  i in range(self.raw*self.column)])
        
        


# In[87]:


# implement
def run_UBQC_sim(runtimes=1,num_bits=20,fibre_len=10**-9,noise_model=None,
               loss_init=0,loss_len=0):
    
    ns.sim_reset()
    # nodes====================================================================

    nodeA = Node("Alice", port_names=["portQA_1","portCA_1","portCA_2"])
    nodeB = Node("Bob"  , port_names=["portQB_1","portCB_1","portCB_2"])

    # processors===============================================================
    
    Alice_processor=sendableQProcessor("processor_A", num_positions=10,
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

    
    
    
    Alice=ClientProtocol(nodeA,Alice_processor,2,5)
    BOb=ServerProtocol(nodeB,Bob_processor,2,5)
    Alice.start()
    BOb.start()
    #ns.logger.setLevel(1)
    ns.sim_run()
    
    #print(Alice.qList)
    
run_UBQC_sim()


# In[75]:


# plot 


# In[ ]:





# In[ ]:





# In[ ]:


# save
    def makeBrickwork(self):
        print("in makeBrickwork")
        
        # assign qubits to a 2D array
        for i in range(self.raw):
            tmp=[]
            #self.brickwork.pop()
            for j in range(self.column):
                tmp.append(self.brickwork.pop(0))
            self.brickwork2D.append(tmp)
        
        print(self.brickwork2D)
        
        '''
        # horizontal Cz
        for i in range(self.raw-1):
            for j in range(self.column-1):
                
                self.processor.execute_program(
                    self.QG_S_Cz, self.brickwork2D,i,j,i,j+1)
        '''

