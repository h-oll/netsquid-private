#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import netsquid as ns
from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components import QuantumMemory
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.components.models.delaymodels import FibreDelayModel

from netsquid.nodes.node import Node
from netsquid.protocols import NodeProtocol

from netsquid.qubits import create_qubits
from netsquid.qubits.operators import *
from netsquid.qubits.qubitapi import *
from netsquid.qubits.operators import Operator

from QuantumTeleportation import *

from random import randint


# In[2]:


def logical_xor(str1, str2):
    return bool(str1) ^ bool(str2) 



def createProcessorAT(name='myProcessor',num_positions=10,memNoiseModel=None,processorNoiseModel=None):
    myProcessor=QuantumProcessor(name, num_positions=num_positions,
        mem_noise_models=memNoiseModel, phys_instructions=[
        PhysicalInstruction(INSTR_X, duration=1  , q_noise_model=processorNoiseModel),
        PhysicalInstruction(INSTR_Z, duration=1  , q_noise_model=processorNoiseModel),
        PhysicalInstruction(INSTR_H, duration=1  , q_noise_model=processorNoiseModel),
        PhysicalInstruction(INSTR_CNOT,duration=1, q_noise_model=processorNoiseModel),
        PhysicalInstruction(INSTR_CZ,duration=10 , q_noise_model=processorNoiseModel),
        PhysicalInstruction(INSTR_MEASURE, duration=10  , q_noise_model=processorNoiseModel, parallel=True),
        PhysicalInstruction(INSTR_MEASURE_X, duration=10, q_noise_model=processorNoiseModel, parallel=True)])
    
    return myProcessor



# In[3]:


'''
Anonymous Transmission in W-state

Star network
Center node

'''

class AT_Wstate_center(NodeProtocol):

    
    def __init__(self,node,processor,numNode): 
        super().__init__()
        self.numNode=numNode
        self.node=node
        self.processor=processor
        
        
        

        
        
    def run(self):
        print("center run~")
    


# In[4]:


'''
Anonymous Transmission in W-state

Star network
Side nodes

'''

class AT_Wstate_side(NodeProtocol):
    
    def __init__(self,node,processor,sender=False,receiver=False): 
        super().__init__()
        self.node=node
        self.processor=processor
        self.sender=sender
        self.receiver=receiver
        
        
        
    def run(self):
        print("side run~")


# In[5]:


def run_AT_sim(numNodes=3,fibre_len=10**-9,processorNoiseModel=None,memNoiseMmodel=None,loss_init=0,loss_len=0
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
        sideProcessorList.append(createProcessorAT(name="myProcessor_"+str(i)))
        
        ### nodes=====================================================================
        sideNodeList.append(Node("node_"+str(i), port_names=["portQSide_"+str(i)]))
        centerPortList.append("PortQCenter_"+str(i))
        
        ### channels==================================================================
        channelList.append(QuantumChannel("QChannel_Center->Side_"+str(i),delay=10,length=fibre_len
            ,models={"quantum_loss_model":
            FibreLossModel(p_loss_init=loss_init,p_loss_length=loss_len, rng=None),
            "delay_model": FibreDelayModel(c=QChV)}))
        
        ### record port list for center node
        centerPortList.append("PortQCenter_"+str(i))

    ## create center component
    CenterNode=Node("CenterNode", port_names=centerPortList)
    CenterProcessor=createProcessorAT(name="myProcessor_center")
    
    ## connect==================================================================
    for i in range(numNodes):    
        CenterNode.connect_to(sideNodeList[i], channelList[i],
            local_port_name =CenterNode.ports["PortQCenter_"+str(i)].name,
            remote_port_name=sideNodeList[i].ports["portQSide_"+str(i)].name)
    
    
    # create protocol object
    myProtocol_center = AT_Wstate_center(CenterNode,CenterProcessor,numNodes)
    myProtocol_sideList=[]
    ## create side protocol
    for i in range(numNodes):
        if i==senderID:
            # create sender
            myProtocol_sideList.append(AT_Wstate_side(sideNodeList[i],sideProcessorList[i],sender=True))
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
    


# In[6]:


#test
run_AT_sim(numNodes=3,fibre_len=10**-9
    ,processorNoiseModel=None,memNoiseMmodel=None,loss_init=0,loss_len=0)




        

