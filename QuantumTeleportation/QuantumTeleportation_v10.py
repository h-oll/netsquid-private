#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.protocols import NodeProtocol
from netsquid.qubits.operators import X,H,Z,CNOT
from netsquid.components  import ClassicalFibre,QuantumFibre

from netsquid.qubits.qubitapi import *
from netsquid.qubits.qformalism import *

from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.components.models.qerrormodels import *
from random import randint
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components import QSource,Clock
from netsquid.components.qsource import SourceStatus

from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.components.models.delaymodels import FibreDelayModel


# In[48]:


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




class TP_SenderTeleport(QuantumProgram):
    
    def __init__(self):
        super().__init__()
        
    def program(self):
        
        # EPR-like        
        self.apply(INSTR_CNOT, [0, 1])
        self.apply(INSTR_H, 0) 
        
        self.apply(INSTR_MEASURE,qubit_indices=0, output_key='0',physical=True) # measure the origin state
        self.apply(INSTR_MEASURE,qubit_indices=1, output_key='1',physical=True) # measure the epr1
        
        yield self.run(parallel=False)

        
        
        
class TP_ReceiverAdjust(QuantumProgram):
    
    def __init__(self,adjBase):
        super().__init__()
        self.adjBase=adjBase
        
        
    def program(self):
        
        if self.adjBase[0]==1:
            self.apply(INSTR_Z, 0)  
        
        if self.adjBase[1]==1:
            self.apply(INSTR_X, 0)
            

        
        yield self.run(parallel=False)
        
        

class QG_measure(QuantumProgram):
    
    def __init__(self,basisList):
        self.basisList=basisList
        super().__init__()


    def program(self):   
        for i in range(len(self.basisList)):
            if self.basisList[i] == 0:                  # standard basis,  0 case
                self.apply(INSTR_MEASURE, 
                    qubit_indices=i, output_key=str(i),physical=True) 
            else:                                      # Hadamard basis,  1 case
                self.apply(INSTR_MEASURE_X, 
                    qubit_indices=i, output_key=str(i),physical=True) 
        yield self.run(parallel=False)

        
def ProgramFail():
    print("A programe failed!!")


# In[49]:


class QuantumTeleportationSender(NodeProtocol):
    
    def __init__(self,node,processor,SendQubit,EPR_1,portNames=["portC_Sender"]): 
        super().__init__()
        self.node=node
        self.processor=processor
        self.SendQubit=SendQubit
        self.EPR_1=EPR_1
        self.measureRes=None
        self.portNameCS1=portNames[0]
        
        self.processor.put([SendQubit,EPR_1])
        
        
        
    def run(self):
        
        # Entangle the two qubits and measure
        self.myTP_SenderTeleport=TP_SenderTeleport()
        self.processor.execute_program(self.myTP_SenderTeleport,qubit_mapping=[0,1])
        self.processor.set_program_done_callback(self.getPGoutput_Sender,once=True)
        self.processor.set_program_fail_callback(ProgramFail,once=True)
        
        yield self.await_program(processor=self.processor)
        
        # Send results to Receiver
        self.node.ports[self.portNameCS1].tx_output(self.measureRes)
        
        
        
    def getPGoutput_Sender(self):
        self.measureRes=getPGoutput(self.myTP_SenderTeleport)


# In[50]:


class QuantumTeleportationReceiver(NodeProtocol):
    
    def __init__(self,node,processor,EPR_2,portNames=["portC_Receiver"]): 
        super().__init__()
        self.node=node
        self.processor=processor
        self.resultQubit=EPR_2
        self.portNameCR1=portNames[0]
        #set_qstate_formalism(QFormalism.DM)
        
        self.processor.put(self.resultQubit)
        
    def run(self):
        
        port=self.node.ports[self.portNameCR1]
        yield self.await_port_input(port)
        res=port.rx_input().items
        #print("R get results:", res)
        
        # edit EPR2 according to res
        self.myTP_ReceiverAdjust=TP_ReceiverAdjust(res)
        self.processor.execute_program(self.myTP_ReceiverAdjust,qubit_mapping=[0])
        self.processor.set_program_done_callback(self.show_state,once=True)
        self.processor.set_program_fail_callback(ProgramFail,once=True)
        
    def show_state(self):
        set_qstate_formalism(QFormalism.DM)
        tmp=self.processor.pop(0)[0]
        print("final state:\n",tmp.qstate.dm)


# In[51]:


def run_Teleport_sim(runtimes=1,fibre_len=10**-9,memNoiseMmodel=None,processorNoiseModel=None
               ,loss_init=0,loss_len=0,QChV=3*10**-4,CChV=3*10**-4):
    
    
    
    for i in range(runtimes): 
        
        ns.sim_reset()

        # nodes====================================================================

        nodeSender   = Node("SenderNode"    , port_names=["portC_Sender"])
        nodeReceiver = Node("ReceiverNode"  , port_names=["portC_Receiver"])

        # processors===============================================================
        processorSender=QuantumProcessor("processorSender", num_positions=10,
            mem_noise_models=memNoiseMmodel, phys_instructions=[
            PhysicalInstruction(INSTR_X, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_H, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_CNOT,duration=10,q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_MEASURE, duration=10,q_noise_model=processorNoiseModel, parallel=False)])


        processorReceiver=QuantumProcessor("processorReceiver", num_positions=10,
            mem_noise_models=memNoiseMmodel, phys_instructions=[
            PhysicalInstruction(INSTR_X, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_H, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_CNOT,duration=10,q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_MEASURE, duration=10,q_noise_model=processorNoiseModel, parallel=False)])


        # channels==================================================================
        
        MyCChannel = ClassicalChannel("CChannel_S->R",delay=0
            ,length=fibre_len)

        nodeSender.connect_to(nodeReceiver, MyCChannel,
                            local_port_name="portC_Sender", remote_port_name="portC_Receiver")

        
        
        # make an EPR pair and origin state
        oriQubit,epr1,epr2=create_qubits(3)
        operate(epr1, H)
        operate([epr1, epr2], CNOT)
        
        
        myQT_Sender = QuantumTeleportationSender(node=nodeSender,
            processor=processorSender,SendQubit=oriQubit,EPR_1=epr1)
        myQT_Receiver = QuantumTeleportationReceiver(node=nodeReceiver,
            processor=processorReceiver,EPR_2=epr2)
        
        myQT_Receiver.start()
        myQT_Sender.start()
        #ns.logger.setLevel(1)
        stats = ns.sim_run()
        
        

    return 0


# In[52]:


run_Teleport_sim()


# In[ ]:




