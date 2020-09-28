#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import netsquid as ns

from netsquid.nodes.node import Node
from netsquid.protocols import NodeProtocol

from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.components.models import  FibreDelayModel
from netsquid.components.models.qerrormodels import *
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components import QSource,Clock
from netsquid.components.qsource import SourceStatus

from netsquid.qubits.operators import Operator,create_rotation_op
from netsquid.qubits.qformalism import *
from random import randint


# In[2]:


# General functions

# Z Rotation operators 
theta = np.pi/8
#print(theta)
# 8 types of rotations
# R0
R22=create_rotation_op(   theta/2, rotation_axis=(0, 0, 1))
R45=create_rotation_op( 2*theta/2, rotation_axis=(0, 0, 1))
R67=create_rotation_op( 3*theta/2, rotation_axis=(0, 0, 1))
R90=create_rotation_op( 4*theta/2, rotation_axis=(0, 0, 1))
R112=create_rotation_op(5*theta/2, rotation_axis=(0, 0, 1))
R135=create_rotation_op(6*theta/2, rotation_axis=(0, 0, 1))
R157=create_rotation_op(7*theta/2, rotation_axis=(0, 0, 1))


INSTR_R22 = IGate('Z Rotated 22.5',operator=R22)
INSTR_R45 = IGate('Z Rotated 45'  ,operator=R45)
INSTR_R67 = IGate('Z Rotated 67.5',operator=R67)
INSTR_R90 = IGate('Z Rotated 90'    ,operator=R90)
INSTR_R112 = IGate('Z Rotated 112.5',operator=R112)
INSTR_R135 = IGate('Z Rotated 135'  ,operator=R135)
INSTR_R157 = IGate('Z Rotated 157.5',operator=R157)

INSTR_Rv22 = IGate('Z Rotated -22.5',operator=R22.inv)
INSTR_Rv45 = IGate('Z Rotated -45'  ,operator=R45.inv)
INSTR_Rv67 = IGate('Z Rotated -67.5',operator=R67.inv)
INSTR_Rv90 = IGate('Z Rotated -90'    ,operator=R90.inv)
INSTR_Rv112 = IGate('Z Rotated -112.5',operator=R112.inv)
INSTR_Rv135 = IGate('Z Rotated -135'  ,operator=R135.inv)
INSTR_Rv157 = IGate('Z Rotated -157.5',operator=R157.inv)


# In[3]:


# Quantum programs
class PrepareEPRpairs(QuantumProgram):
    
    def __init__(self,pairs=1):
        self.pairs=pairs
        super().__init__()
        
    def program(self):
        qList_idx=self.get_qubit_indices(2*self.pairs)
        # create multiEPR
        for i in range(2*self.pairs):
            if i%2==0:                           # List A case
                self.apply(INSTR_H, qList_idx[i])
            else:                                # List B case
                self.apply(INSTR_CNOT, [qList_idx[i-1], qList_idx[i]])
        yield self.run(parallel=False)


'''
Measure the qubits hold by this processor by basisList.
input:
    basisList:list of int(0/1): indecate measurement basis

'''
class QMeasure(QuantumProgram):
    def __init__(self,basisList):
        self.basisList=basisList
        super().__init__()

    def program(self):
        print("in QMeasure")
        for i in range(0,len(self.basisList)):
            if self.basisList[int(i/2)] == 0:  # basisList 0:Z  , 1:X        
                self.apply(INSTR_MEASURE, 
                    qubit_indices=i, output_key=str(i),physical=True) 
            else:                              
                self.apply(INSTR_MEASURE_X, 
                    qubit_indices=i, output_key=str(i),physical=True)

        yield self.run(parallel=False)


'''
input:
    positionInx:int: Index in Qmem to measure.
    angleInx:int([0,7]): Index indecating measurement angle along Z-axis.
output:
'''
class AngleMeasure(QuantumProgram):
    def __init__(self,positionInx,angleInx):
        self.positionInx=positionInx
        self.angleInx=angleInx
        super().__init__()

    def program(self):
        print("in AngleMeasure")
        print("self.positionInx",self.positionInx)
        print("self.angleInx",self.angleInx)
        if   self.angleInx == 1:
            self.apply(INSTR_R22, self.positionInx)
            self.apply(INSTR_MEASURE,qubit_indices=self.positionInx, output_key="1",physical=True)
            self.apply(INSTR_Rv22, self.positionInx)
        elif self.angleInx == 2:
            self.apply(INSTR_R45, self.positionInx)
            self.apply(INSTR_MEASURE,qubit_indices=self.positionInx, output_key="1",physical=True)
            self.apply(INSTR_Rv45, self.positionInx)
        elif self.angleInx == 3:
            self.apply(INSTR_R67, self.positionInx)
            self.apply(INSTR_MEASURE,qubit_indices=self.positionInx, output_key="1",physical=True)
            self.apply(INSTR_Rv67, self.positionInx)
        elif self.angleInx == 4:
            self.apply(INSTR_R90, self.positionInx)
            self.apply(INSTR_MEASURE,qubit_indices=self.positionInx, output_key="1",physical=True)
            self.apply(INSTR_Rv90, self.positionInx)
        elif self.angleInx == 5:
            self.apply(INSTR_R112, self.positionInx)
            self.apply(INSTR_MEASURE,qubit_indices=self.positionInx, output_key="1",physical=True)
            self.apply(INSTR_Rv112, self.positionInx)
        elif self.angleInx == 6:
            self.apply(INSTR_R135, self.positionInx)
            self.apply(INSTR_MEASURE,qubit_indices=self.positionInx, output_key="1",physical=True)
            self.apply(INSTR_Rv135, self.positionInx)
        elif self.angleInx == 7:
            self.apply(INSTR_R157, self.positionInx)
            self.apply(INSTR_MEASURE,qubit_indices=self.positionInx, output_key="1",physical=True)
            self.apply(INSTR_Rv157, self.positionInx)
        else:  # self.angleInx == 0
            self.apply(INSTR_MEASURE,qubit_indices=self.positionInx, output_key="1",physical=True)
        yield self.run(parallel=False)
        
'''
input:
    Pg: A quantum program (QuantumProgram)
output:
    resList: A list of outputs from the given quantum program, 
    also sorted by key.(list of int)
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


# In[4]:


# server protocol
class ProtocolServer(NodeProtocol):
    
    def __init__(self,node,processor,port_names=["portQS_1","portCS_1","portCS_2"],realRound=5):
        super().__init__()
        self.node=node
        self.processor=processor
        self.portNameQ1=port_names[0]
        self.portNameC1=port_names[1]
        self.portNameC2=port_names[2]
        self.sourceQList = []
        self.port_output = []
        self.realRound = realRound
            
    def S_genQubits(self,num):
        #generat qubits from source
        A_Source = QSource("S_source") 
        A_Source.ports["qout0"].bind_output_handler(self.store_output_from_port)
        A_Source.status = SourceStatus.EXTERNAL
        
        #set clock
        clock = Clock("clock", frequency=1e9, max_ticks=num)
        clock.ports["cout"].connect(A_Source.ports["trigger"])
        clock.start()
        
        
    def store_output_from_port(self,message):
        self.port_output.append(message.items[0])
        if len(self.port_output)==2:
            print("store_output_from_port:",self.port_output)
            self.processor.put(qubits=self.port_output)
            
            # do H CNOT operation
            # PrepareEPRpairs
            prepareEPRpairs=PrepareEPRpairs(1)
            self.processor.execute_program(
                prepareEPRpairs,qubit_mapping=[i for  i in range(0, 2)])
    
    
    def S_sendEPR(self):
        #print("S_sendEPR")
        payload=self.processor.pop(1)
        self.node.ports[self.portNameQ1].tx_output(payload)        
        
    
    
    def run(self):
        print("server on")
        port = self.node.ports["portCS_1"]
        
        #receive qubits from client
        yield self.await_port_input(port)
        rounds = port.rx_input().items
        print("Server received rounds:",rounds)
        
        # send half of an EPRpair to client
        
        # gen 2 qubits
        self.S_genQubits(2)
        
        yield self.await_program(processor=self.processor)
        self.S_sendEPR()
        
        
        


# In[5]:


# client protocol
class ProtocolClient(NodeProtocol):
    
    def __init__(self,node,processor,port_names=["portQC_1","portCC_1","portCC_2"],maxRounds=10):
        super().__init__()
        self.node=node
        self.processor=processor
        self.portNameQ1=port_names[0]
        self.portNameC1=port_names[1]
        self.portNameC2=port_names[2]
        self.maxRounds=maxRounds
        self.d=randint(1,2)
        self.z2=None
        self.theta2=None
        self.r2=None
        self.p2=None
    
    def run(self):
        print("client on")
        testsms=2
        self.node.ports[self.portNameC1].tx_output(testsms)
        
        
        port = self.node.ports["portQC_1"]
        #receive qubits from client
        yield self.await_port_input(port)
        
        aEPR=port.rx_input().items
        print("B received qubits:",aEPR)
        self.processor.put(aEPR)
        
        if self.d == 2 :
            # measure the only qubit in Z basis
            print("case d=2")
            myQMeasure=QMeasure([0]) 
            self.processor.execute_program(myQMeasure,qubit_mapping=[0])
            self.processor.set_program_done_callback(self.myGetPGoutput,myQMeasure,once=True) #not working
        
            yield self.await_program(processor=self.processor)
            #send ACK
            self.node.ports["portCC_1"].tx_output("ACK")
            
            
        else:
            print("case d=1")
            self.theta2=randint(0,7)
            self.r2=randint(0,1)
            # measure by theta2
            myAngleMeasure=AngleMeasure(0,self.theta2) # first qubit
            self.processor.execute_program(myAngleMeasure,qubit_mapping=[0])
            self.processor.set_program_done_callback(self.myGetPGoutput,myAngleMeasure,once=True)
            self.processor.set_program_fail_callback(self.ProgramFail,once=True)
            yield self.await_program(processor=self.processor)
            #send ACK
            self.node.ports["portCC_1"].tx_output("ACK")
            
            
        
    def myGetPGoutput(self,QG):
        if self.d == 2 :
            self.z2 = getPGoutput(QG)
            print("self.z2=",self.z2)
        elif self.d == 1 :
            self.p2 = getPGoutput(QG)
            print("self.p2=",self.p2)
        else:
            print("error")
            
    def ProgramFail(self):
        print("programe failed!!")


# In[6]:


# implementation & hardware configure
def run_UBQC_sim(runtimes=1,num_bits=20,fibre_len=10**-9,noise_model=None,
               loss_init=0,loss_len=0):
    
    for i in range(runtimes): 
        
        ns.sim_reset()

        # nodes====================================================================

        nodeServer = Node("Server", port_names=["portQS_1","portCS_1","portCS_2"])
        nodeClient = Node("Client"  , port_names=["portQC_1","portCC_1","portCC_2"])

        # processors===============================================================
        noise_model=None #try
        
        
        processorServer=QuantumProcessor("processorServer", num_positions=100,
            mem_noise_models=None, phys_instructions=[
            PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
            PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_CNOT,duration=1,q_noise_model=noise_model),
            PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R22, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R45, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R67, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R90, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R112, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R135, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R157, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv22, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv45, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv67, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv90, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv112, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv135, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv157, duration=1, parallel=True)])
        
        
        
        processorClient=QuantumProcessor("processorClient", num_positions=100,
            mem_noise_models=None, phys_instructions=[
            PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
            PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_CNOT,duration=1,q_noise_model=noise_model),
            PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R22, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R45, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R67, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R90, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R112, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R135, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R157, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv22, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv45, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv67, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv90, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv112, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv135, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv157, duration=1, parallel=True)])


        # channels==================================================================
        
        #FibreDelayModel()
    
        MyQChannel=QuantumChannel("QChannel_S->C",delay=0
            ,length=fibre_len,models=noise_model)
        
        
        nodeServer.connect_to(nodeClient, MyQChannel,
            local_port_name =nodeServer.ports["portQS_1"].name,
            remote_port_name=nodeClient.ports["portQC_1"].name)
        

        MyCChannel = ClassicalChannel("CChannel_C->S",delay=0
            ,length=fibre_len)
        MyCChannel2= ClassicalChannel("CChannel_S->C",delay=0
            ,length=fibre_len)
        

        nodeClient.connect_to(nodeServer, MyCChannel,
                            local_port_name="portCC_1", remote_port_name="portCS_1")
        nodeServer.connect_to(nodeClient, MyCChannel2,
                            local_port_name="portCS_2", remote_port_name="portCC_2")


        protocolServer = ProtocolServer(nodeServer,processorServer)
        protocolClient = ProtocolClient(nodeClient,processorClient)
        protocolServer.start()
        protocolClient.start()
        #ns.logger.setLevel(1)
        stats = ns.sim_run()


# In[7]:


# test
run_UBQC_sim()


# In[ ]:




'''
R22 = Operator("R22",     np.dot(np.cos(   theta/2.),Iarr) - 1.j * np.dot(np.sin(   theta/2.) , Zarr)) 
R45 = Operator("R45",     np.dot(np.cos(2.*theta/2.),Iarr) - 1.j * np.dot(np.sin(2.*theta/2.) , Zarr)) 
R67 = Operator("R67",     np.dot(np.cos(3.*theta/2.),Iarr) - 1.j * np.dot(np.sin(3.*theta/2.) , Zarr)) 
R90 = Operator("R90",     np.dot(np.cos(4.*theta/2.),Iarr) - 1.j * np.dot(np.sin(4.*theta/2.) , Zarr)) 
R112 = Operator("R112",   np.dot(np.cos(5.*theta/2.),Iarr) - 1.j * np.dot(np.sin(5.*theta/2.) , Zarr)) 
R135 = Operator("R135",   np.dot(np.cos(6.*theta/2.),Iarr) - 1.j * np.dot(np.sin(6.*theta/2.) , Zarr)) 
R157 = Operator("R157",   np.dot(np.cos(7.*theta/2.),Iarr) - 1.j * np.dot(np.sin(7.*theta/2.) , Zarr)) 
'''

