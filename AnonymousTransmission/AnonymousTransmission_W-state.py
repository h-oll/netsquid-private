#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import netsquid as ns
from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components import QuantumMemory,QSource,Clock
from netsquid.components.qchannel import QuantumChannel 
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.components.qsource import SourceStatus
from netsquid.components.qprogram import *

from netsquid.nodes.node import Node
from netsquid.protocols import NodeProtocol

from netsquid.qubits import create_qubits
from netsquid.qubits.operators import *
from netsquid.qubits.qubitapi import *
from netsquid.qubits.operators import Operator

from QuantumTeleportation import *

from random import randint


# In[2]:


NToffoli_matrix = [[0, 1, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1]]

Operator_NToffoli = Operator(name='Operator_NToffoli', matrix=NToffoli_matrix)
INSTR_NToffoli = IGate('INSTR_NToffoli',operator=Operator_NToffoli)


# In[3]:


def logical_xor(str1, str2):
    return bool(str1) ^ bool(str2) 



def createProcessorAT(name='defaultProcessor',num_positions=4,memNoiseModel=None,processorNoiseModel=None):


    myProcessor=QuantumProcessor(name, num_positions=num_positions,
        mem_noise_models=memNoiseModel, phys_instructions=[
        PhysicalInstruction(INSTR_X, duration=1  , q_noise_model=processorNoiseModel),
        PhysicalInstruction(INSTR_Z, duration=1  , q_noise_model=processorNoiseModel),
        PhysicalInstruction(INSTR_H, duration=1  , q_noise_model=processorNoiseModel),
        PhysicalInstruction(INSTR_CNOT,duration=1, q_noise_model=processorNoiseModel),
        PhysicalInstruction(INSTR_CZ,duration=10 , q_noise_model=processorNoiseModel),
        PhysicalInstruction(INSTR_MEASURE, duration=10  , q_noise_model=processorNoiseModel, parallel=False),
        PhysicalInstruction(INSTR_MEASURE_X, duration=10, q_noise_model=processorNoiseModel, parallel=False),
        PhysicalInstruction(INSTR_TOFFOLI, duration=10, q_noise_model=processorNoiseModel, parallel=False),
        PhysicalInstruction(INSTR_NToffoli, duration=10)])
    
    return myProcessor


class makeWstate(QuantumProgram):
    
    def __init__(self,numQubits=4):
        self.numQubits=numQubits
        super().__init__()
        
    def program(self):
        if self.numQubits%4==0:
            self.apply(INSTR_H, 2)
            self.apply(INSTR_H, 3)
            self.apply(INSTR_NToffoli, [3,2,1])
            self.apply(INSTR_TOFFOLI,qubit_indices=[3,2,0],physical=True) 
            self.apply(INSTR_CNOT, [0, 2])
            self.apply(INSTR_CNOT, [0, 3])
        else:
            print("numbers of qubits should be a multiple of 4")
        
        yield self.run(parallel=False)
        

def ProgramFail(info):
    print(info)
    print("programe failed!!")    
    
    
    
    
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


'''
General measurement function.
input:
    basisList: List of int, 0 means standard basis, others means Hadamard basis
'''
class QG_measure(QuantumProgram):
    def __init__(self,basisList):
        self.basisList=basisList
        super().__init__()
        
    def program(self):
        for i in range(len(self.basisList)):
            #print("QG_measure ",i)
            if self.basisList[i] == 0:                  # standard basis, case 0
                self.apply(INSTR_MEASURE, 
                    qubit_indices=i, output_key=str(i),physical=True) 
            else:                                       # Hadamard basis, case 1
                self.apply(INSTR_MEASURE_X, 
                    qubit_indices=i, output_key=str(i),physical=True) 
        yield self.run(parallel=False)


# In[4]:


'''
Anonymous Transmission in W-state

Star network
Center node

'''

class AT_Wstate_center(NodeProtocol):

    
    def __init__(self,node,processor,numNode,portQlist): 
        super().__init__()
        self.numNode=numNode
        self.node=node
        self.processor=processor
        self.portQlist=portQlist
        
        self.sourceQList=[]
        
        
        self.C_Source = QSource("center_source"
            ,status=SourceStatus.EXTERNAL) # enable frequency
        self.C_Source.ports["qout0"].bind_output_handler(self.storeSourceOutput)
        
        
        
    def run(self):
        print("center run~")
        
        self.C_genQubits(self.numNode) # make W state too
        
        yield self.await_program(processor=self.processor)
        print("w state finished")
        
        self.C_sendWstate()
        
        #yield self.await_program(processor=self.processor)
        #print("qubit gen finished")
        
        
    def storeSourceOutput(self,qubit):
        self.sourceQList.append(qubit.items[0])
        if len(self.sourceQList)==self.numNode:
            print("sourceQList:",self.sourceQList,"putting in Qmem")
            self.processor.put(qubits=self.sourceQList)
            
            myMakeWstate = makeWstate(self.numNode)
            self.processor.execute_program(myMakeWstate,qubit_mapping=[i for i in range(self.numNode)])
            self.processor.set_program_fail_callback(ProgramFail,once=True)
            
            
            
    def C_genQubits(self,num,freq=1e9):
        #set clock
        clock = Clock("clock", frequency=freq, max_ticks=num)
        try:
            clock.ports["cout"].connect(self.C_Source.ports["trigger"])
        except:
            pass    #print("alread connected") 
            
        clock.start()
    
    
    
    def C_sendWstate(self):
        print("C_sendWstate")
        
        for i in reversed(range(self.numNode)):
            payload=self.processor.pop(i)
            #print("i:",i," payload:",payload)
            #print("portQlist[i]: ",self.portQlist[i])
            self.node.ports[self.portQlist[i]].tx_output(payload)
            


# In[5]:


'''
Anonymous Transmission in W-state

Star network
Side nodes

'''

class AT_Wstate_side(NodeProtocol):
    
    def __init__(self,node,processor,sender=False,receiver=False,portQlist=["portQside"]): 
        super().__init__()
        self.node=node
        self.processor=processor
        self.sender=sender
        self.receiver=receiver
        
        self.portQlist=portQlist
        self.wStateResult=None
        
        
    def run(self):
        print(self.processor.name)
        self.showIdentity()
        
        # Side receive a qubit from Center
        port=self.node.ports[self.portQlist[0]]
        yield self.await_port_input(port)
        wQubit = port.rx_input().items
        print("I received:",wQubit)
        self.processor.put(wQubit[0])
        
        # Side measures the qubit in standard basis if not sender or receiver.
        if (self.sender==False) and (self.receiver==False) :
            print(self.processor.name)
            self.myQG_measure = QG_measure([0])
            self.processor.execute_program(self.myQG_measure,qubit_mapping=[0])
            self.processor.set_program_done_callback(self.S_getPGoutput,once=True)
            self.processor.set_program_fail_callback(ProgramFail,info=self.processor.name,once=True)
        else:
            print("else case")
        
        yield self.await_program(processor=self.processor)
        print("self.wStateResult: ",self.wStateResult)
        
        
        
        
    def showIdentity(self):
        if self.sender==True:
            print("I am sender")
        elif self.receiver==True:
            print("I am receiver")
        else:
            print("I am normal side node")
            

    def S_getPGoutput(self):
        self.wStateResult=getPGoutput(self.myQG_measure)
        #print("wStateResult:",self.wStateResult)


# In[6]:


def run_AT_sim(numNodes=4,fibre_len=10**-9,processorNoiseModel=None,memNoiseMmodel=None,loss_init=0,loss_len=0
              ,QChV=3*10**-4):
    
    # initialize
    ns.sim_reset()
    
    sideProcessorList=[]
    sideNodeList=[]
    centerPortList=[]
    channelList=[]
    
    senderID=randint(0,numNodes-1)
    receiverID=randint(0,numNodes-1)
    while receiverID==senderID:
        receiverID=randint(0,numNodes-1)
    
    
    # build star network hardware components
    ## create side components
    for i in range(numNodes):
        ### processors================================================================
        sideProcessorList.append(createProcessorAT(name="ProcessorSide_"+str(i)))
        
        ### nodes=====================================================================
        sideNodeList.append(Node("node_"+str(i), port_names=["portQside"]))
        
        ### channels==================================================================
        channelList.append(QuantumChannel("QChannel_Center->Side_"+str(i),delay=10,length=fibre_len
            ,models={"quantum_loss_model":
            FibreLossModel(p_loss_init=loss_init,p_loss_length=loss_len, rng=None),
            "delay_model": FibreDelayModel(c=QChV)}))
        
        ### record port list for center node
        centerPortList.append("PortQCenter_"+str(i))

        
        
    ## create center component
    CenterNode=Node("CenterNode", port_names=centerPortList)
    CenterProcessor=createProcessorAT(name="ProcessorCenter")
    
    ## connect==================================================================
    for i in range(numNodes):    
        CenterNode.connect_to(sideNodeList[i], channelList[i],
            local_port_name =CenterNode.ports["PortQCenter_"+str(i)].name,
            remote_port_name=sideNodeList[i].ports["portQside"].name)
    
    
    # create protocol object
    myProtocol_center = AT_Wstate_center(CenterNode,CenterProcessor,numNodes,portQlist=centerPortList)
    myProtocol_sideList=[]
    ## create side protocol
    for i in range(numNodes):
        if i==senderID:
            # create sender
            myProtocol_sideList.append(AT_Wstate_side(sideNodeList[i],sideProcessorList[i],  sender=True))
        elif i==receiverID:
            # create receiver
            myProtocol_sideList.append(AT_Wstate_side(sideNodeList[i],sideProcessorList[i],receiver=True))
        else:
            # create normal side node
            myProtocol_sideList.append(AT_Wstate_side(sideNodeList[i],sideProcessorList[i]))
        
    
    
    for sideProtocols in myProtocol_sideList:
        sideProtocols.start()
        
    myProtocol_center.start()
    
    
    #ns.logger.setLevel(1)
    stats = ns.sim_run()
    




#test
run_AT_sim(numNodes=4,fibre_len=10**-9
    ,processorNoiseModel=None,memNoiseMmodel=None,loss_init=0,loss_len=0)





        

