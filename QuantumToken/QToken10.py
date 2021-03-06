#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.protocols import NodeProtocol
from netsquid.qubits.operators import X,H,Z
from netsquid.qubits.qformalism import *
from netsquid.pydynaa import Entity,EventHandler,EventType
from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.components.models.qerrormodels import *
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components import QSource,Clock
from netsquid.components.qsource import SourceStatus
from netsquid.components.models.qerrormodels import FibreLossModel

from random import randint


# In[2]:



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
input:
    basisInxList: Oroginal sets of qubits:(list of N)
        0:(0,+)  1:(0,-)  2:(1,+)  3:(1,-)
        4:(+,0)  5:(+,1)  6:(-,0)  7:(-,1)
    randMeas:(int 0/1)
        0: standard basis   1: H basis
    locRes:(list of 2*N)
        received measurement to check
output:
    res: 
        the persentage of passed qubits among all qubits.
'''
def TokenCheck(basisInxList,randMeas,locRes):
    failCount=0
    if randMeas==0:
        for i in range(len(basisInxList)):
            if basisInxList[i]<=1 and locRes[2*i]==0:
                pass
            elif basisInxList[i]<=3 and locRes[2*i]==1:
                pass
            elif basisInxList[i]%2==0 and locRes[2*i+1]==0:
                pass
            elif basisInxList[i]%2==1 and locRes[2*i+1]==1:
                pass
            else:
                #print("false case1:",i)
                failCount+=1
    else: # randMeas==1:
        for i in range(len(basisInxList)):
            if basisInxList[i]>=6 and locRes[2*i]==1:
                pass
            elif basisInxList[i]>=4 and locRes[2*i]==0:
                pass
            elif basisInxList[i]%2==0 and locRes[2*i+1]==0:
                pass
            elif basisInxList[i]%2==1 and locRes[2*i+1]==1:
                pass
            else:
                #print("false case2:",i)
                failCount+=1
    
    return (len(basisInxList)-failCount)/len(basisInxList)


# class of quantum program
class QG_B_qPrepare(QuantumProgram):
    def __init__(self,num_bits,stateInxList):
        self.num_bits=num_bits
        self.stateInxList=stateInxList
        super().__init__()
        
    def program(self):
        qList_idx=self.get_qubit_indices(2*self.num_bits)
        '''
        0:(0,+)  1:(0,-)  2:(1,+)  3:(1,-)
        4:(+,0)  5:(+,1)  6:(-,0)  7:(-,1)
        '''
        for i in range(self.num_bits):
            if self.stateInxList[i]==0:                           
                self.apply(INSTR_H, qList_idx[2*i+1])
            elif self.stateInxList[i]==1:                                
                self.apply(INSTR_X, qList_idx[2*i+1])
                self.apply(INSTR_H, qList_idx[2*i+1])
            elif self.stateInxList[i]==2:                                
                self.apply(INSTR_X, qList_idx[2*i])
                self.apply(INSTR_H, qList_idx[2*i+1])
            elif self.stateInxList[i]==3:                                
                self.apply(INSTR_X, qList_idx[2*i])
                self.apply(INSTR_X, qList_idx[2*i+1])
                self.apply(INSTR_H, qList_idx[2*i+1])
            elif self.stateInxList[i]==4:                                
                self.apply(INSTR_H, qList_idx[2*i])
            elif self.stateInxList[i]==5:                                
                self.apply(INSTR_H, qList_idx[2*i])
                self.apply(INSTR_X, qList_idx[2*i+1])
            elif self.stateInxList[i]==6:                                
                self.apply(INSTR_X, qList_idx[2*i])
                self.apply(INSTR_H, qList_idx[2*i])
            else : #"stateInx==7"
                self.apply(INSTR_X, qList_idx[2*i])
                self.apply(INSTR_H, qList_idx[2*i])
                self.apply(INSTR_X, qList_idx[2*i+1])
                
        yield self.run(parallel=False)

        
        

class QG_A_measure(QuantumProgram):
    def __init__(self,basisList,num_bits):
        self.basisList=basisList
        self.num_bits=num_bits
        super().__init__()


    def program(self):   
        for i in range(len(self.basisList)):
            if self.basisList[int(i)] == 0:    # standard basis
                self.apply(INSTR_MEASURE, 
                    qubit_indices=i, output_key=str(i),physical=True) 
            else:                              # 1 case # Hadamard basis
                self.apply(INSTR_MEASURE_X, 
                    qubit_indices=i, output_key=str(i),physical=True) 
        yield self.run(parallel=False)
        
        


# In[3]:


class AliceProtocol(NodeProtocol):
    
    def __init__(self,node,processor,num_bits,waitTime,
                port_names=["portQA_1","portCA_1","portCA_2"]):
        super().__init__()
        self.num_bits=num_bits
        self.node=node
        self.processor=processor
        self.portNameQ1=port_names[0]
        self.portNameC1=port_names[1]
        self.portNameC2=port_names[2]
        self.waitTime=waitTime
        self.tokenQlist = None
        self.loc_mesRes = None
        self.myQG_A_measure = None
        
    # =======================================A run ============================
    def run(self):
        
        # receive qubits from B
        port=self.node.ports[self.portNameQ1]
        yield self.await_port_input(port)
        qubitPairs = port.rx_input().items
        #print("A received qubitPairs=",qubitPairs)
        
        self.processor.put(qubitPairs)
        
        # A keep it for some time
        self.WaitNReqChallenge()
        
        
        port=self.node.ports[self.portNameC2]
        yield self.await_port_input(port)
        
        #print("A received:",port.rx_input().items)
        basis=port.rx_input().items[0]
        #print("basis:",basis)
        
        basisList=[basis for i in range(2*self.num_bits)]
        #print("basisList:",basisList)
        
        
        #print("mem 1 used?  ",self.processor.get_position_used(2*self.num_bits))
        
        self.myQG_A_measure=QG_A_measure(basisList=basisList,num_bits=2*self.num_bits)
        self.processor.execute_program(
            self.myQG_A_measure,qubit_mapping=[i for  i in range(0, 2*self.num_bits)])
        
        
        self.processor.set_program_done_callback(self.A_getPGoutput,once=True)
        yield self.await_program(processor=self.processor)
        #print("self.loc_mesRes",self.loc_mesRes)
        
        self.node.ports[self.portNameC1].tx_output(self.loc_mesRes)
        
        
        port=self.node.ports[self.portNameC2]
        yield self.await_port_input(port)
        Res=port.rx_input().items[0]
        #print("A received result:",Res)
        
        

    def A_getPGoutput(self):
        self.loc_mesRes=getPGoutput(self.myQG_A_measure)
        
        
        
    def WaitNReqChallenge(self):
        
        # schedule waiting for event
        My_waitENVtype = EventType("WAIT_EVENT", "Wait for N nanoseconds")
        self._schedule_after(self.waitTime, My_waitENVtype) 
        self._wait_once(ns.EventHandler(self.ApopMem),entity=self
            ,event_type=My_waitENVtype) 
        
        
        
    def ApopMem(self,event):
        # pop out from qmem and ready for measure
        #self.tokenQlist=self.processor.pop(list(np.arange(len(self.tokenQlist)))) 
        #pop all
        message = "10101"    #use 10101 as request of challenge
        self.node.ports["portCA_1"].tx_output(message)
        
            
        


# In[4]:


class BobProtocol(NodeProtocol):
    
    def __init__(self,node,processor,num_bits,threshold=0.854,
                port_names=["portQB_1","portCB_1","portCB_2"]):
        super().__init__()
        self.num_bits=num_bits
        self.node=node
        self.processor=processor
        self.portNameQ1=port_names[0]
        self.portNameC1=port_names[1]
        self.portNameC2=port_names[2]
        # init value assume that all qubits are lost
        self.sourceQList=[]
        self.basisInxList=[randint(0,7) for i in range(self.num_bits)]
        self.randMeas=randint(0,1) #0:Z basis(standard)   1:X basis(H)
        self.locRes = None
        self.threshold = threshold
        self.successfulRate=None
        
        
        #generat qubits from source
        self.B_Source = QSource("Bank_source"
            ,status=SourceStatus.EXTERNAL) # enable frequency
        self.B_Source.ports["qout0"].bind_output_handler(self.storeSourceOutput)
        
        #print("basisInxList:",self.basisInxList)

    # =======================================B run ============================
    def run(self):
        
        self.B_genQubits(self.num_bits,1e9)
        
        yield self.await_program(processor=self.processor)
        
        self.B_sendQubit()
        
        
        port = self.node.ports[self.portNameC1]
        yield self.await_port_input(port)
        #print(port.rx_input().items)
        reqMes = port.rx_input().items[0]
        if  reqMes == '10101':
    
            # send payload
            #print("send rand measurement!")
            self.node.ports[self.portNameC2].tx_output(self.randMeas)
        else:
            print("req error!")
            print(reqMes)
            
        #print("waiting for result")
        port = self.node.ports[self.portNameC1]
        yield self.await_port_input(port)
        self.locRes = port.rx_input().items
        #print("locRes:",self.locRes)
        
        self.successfulRate=TokenCheck(self.basisInxList,self.randMeas,self.locRes)
        #print("B successfulRate:",self.successfulRate)
        
        # send result to A
        if self.successfulRate > self.threshold :
            # pass
            self.node.ports[self.portNameC2].tx_output(True)
        else:
            # you shall not pass!
            self.node.ports[self.portNameC2].tx_output(False)
            
            

    def B_genQubits(self,num,freq=1e9):
        
        
        #set clock
        clock = Clock("clock", frequency=freq, max_ticks=2*num)
        try:
            clock.ports["cout"].connect(self.B_Source.ports["trigger"])
        except:
            pass
            #print("alread connected") 
            
        clock.start()
        
    def storeSourceOutput(self,qubit):
        self.sourceQList.append(qubit.items[0])
        if len(self.sourceQList)==2*self.num_bits:
            self.processor.put(qubits=self.sourceQList)
            
            inxList=[randint(0,7) for i in range(self.num_bits)]
            
            #print("inxList=",self.basisInxList)
            # apply H detector
            PG_qPrepare=QG_B_qPrepare(num_bits=self.num_bits,stateInxList=self.basisInxList)
            self.processor.execute_program(
                PG_qPrepare,qubit_mapping=[i for  i in range(0, 2*self.num_bits)])
            

            
    def B_sendQubit(self):
        #print("B_sendQubit")
        inx=list(range(2*self.num_bits))
        payload=self.processor.pop(inx)
        self.node.ports[self.portNameQ1].tx_output(payload)
      


# In[5]:


# implementation & hardware configure
def run_QToken_sim(runTimes=1,num_bits=100,fibre_len=0,waitTime=1,
        processNoiseModel=None,memNoiseModel=None,loss_init=0,loss_len=0,threshold=0.854,
        fibreLoss_init=0.2,fibreLoss_len=0.25,QChDelay=1,CChDelay=0):
    
    resList=[]
    
    for i in range(runTimes): 
        
        ns.sim_reset()

        # nodes====================================================================

        nodeA = Node("Alice", port_names=["portQA_1","portCA_1","portCA_2"])
        nodeB = Node("Bob"  , port_names=["portQB_1","portCB_1","portCB_2"])

        # processors===============================================================
        #noise_model=None
        Alice_processor=QuantumProcessor("processor_A", num_positions=3*10**3,
            mem_noise_models=memNoiseModel, phys_instructions=[
            PhysicalInstruction(INSTR_X, duration=1, q_noise_model=processNoiseModel),
            PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=processNoiseModel),
            PhysicalInstruction(INSTR_H, duration=1, q_noise_model=processNoiseModel),
            PhysicalInstruction(INSTR_CNOT,duration=10,q_noise_model=processNoiseModel),
            PhysicalInstruction(INSTR_MEASURE, duration=10,q_noise_model=processNoiseModel, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=10,q_noise_model=processNoiseModel, parallel=True)])


        Bob_processor=QuantumProcessor("processor_B", num_positions=3*10**3,
            mem_noise_models=memNoiseModel, phys_instructions=[
            PhysicalInstruction(INSTR_X, duration=1, q_noise_model=processNoiseModel),
            PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=processNoiseModel),
            PhysicalInstruction(INSTR_H, duration=1, q_noise_model=processNoiseModel),
            PhysicalInstruction(INSTR_CNOT,duration=10,q_noise_model=processNoiseModel),
            PhysicalInstruction(INSTR_MEASURE, duration=10,q_noise_model=processNoiseModel, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=10,q_noise_model=processNoiseModel, parallel=True)])


        # channels==================================================================
        
        MyQChannel=QuantumChannel("QChannel_B->A",delay=QChDelay,length=fibre_len
            ,models={"myFibreLossModel": FibreLossModel(p_loss_init=fibreLoss_init
            ,p_loss_length=fibreLoss_len, rng=None)})
        
        
        nodeB.connect_to(nodeA, MyQChannel,
            local_port_name =nodeB.ports["portQB_1"].name,
            remote_port_name=nodeA.ports["portQA_1"].name)
        
        
        MyCChannel= ClassicalChannel("CChannel_A->B",delay=CChDelay
            ,length=fibre_len)
        MyCChannel2 = ClassicalChannel("CChannel_B->A",delay=CChDelay
            ,length=fibre_len)
        
        
        nodeA.connect_to(nodeB, MyCChannel,
                            local_port_name="portCA_1", remote_port_name="portCB_1")
        nodeB.connect_to(nodeA, MyCChannel2,
                            local_port_name="portCB_2", remote_port_name="portCA_2")
        
        

        Alice_protocol = AliceProtocol(nodeA,Alice_processor,num_bits,waitTime=waitTime)
        Bob_protocol = BobProtocol(nodeB,Bob_processor,num_bits,threshold=threshold)
        Bob_protocol.start()
        Alice_protocol.start()
        #ns.logger.setLevel(1)
        stats = ns.sim_run()
        
        resList.append(Bob_protocol.successfulRate) 
        #print("Bob_protocol.successfulRate:",Bob_protocol.successfulRate)
        
        
    return sum(resList)/len(resList)


# In[ ]:


import matplotlib.pyplot as plt

#threshold doesn't matter in this plot
def QuantumToken_plot():
    y_axis=[]
    x_axis=[]
    runTimes=5
    
    min_storetime=0
    max_storetime=5000
    
    myMemNoise=T1T2NoiseModel(T1=10, T2=1)
    myProcessNoise=DepolarNoiseModel(depolar_rate=50)

    # first curve
    for i in range(min_storetime,max_storetime,100): # from 0 to 10**8 ns
        x_axis.append(i) # relate to unit
        y_axis.append(run_QToken_sim(runTimes=runTimes,num_bits=100,fibre_len=10,waitTime=i
            ,processNoiseModel=myProcessNoise,memNoiseModel=myMemNoise,loss_init=0.5,loss_len=0.2,threshold=0.854
            ,fibreLoss_init=0.5,fibreLoss_len=0.2,QChDelay=1,CChDelay=0)) 

        
    plt.plot(x_axis, y_axis, 'r-',label='fibre length=10')
    
    '''
    y_axis.clear() 
    x_axis.clear()
    
    # second curve
    for i in range(min_storetime,max_storetime,50): # from 0 to 10**8 ns
        x_axis.append(i) # relate to unit
        y_axis.append(run_QToken_sim(runTimes=runTimes,num_bits=100,fibre_len=50,waitTime=i
            ,processNoiseModel=None,memNoiseModel=None,loss_init=0.5,loss_len=0.2,threshold=0.854
            ,fibreLoss_init=0.5,fibreLoss_len=0.2,QChDelay=1,CChDelay=0)) 

        
    plt.plot(x_axis, y_axis, 'b-',label='fibre length=50')
    '''
     
    plt.title('Quantum Token')
    plt.ylabel('average successful rate')
    plt.xlabel('token kept time (ns)') #µ

    plt.legend()
    plt.savefig('QTplotN113.png')
    plt.show()

    

QuantumToken_plot()

