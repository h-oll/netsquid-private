#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.protocols import NodeProtocol
from netsquid.qubits.operators import X,H,Z
from netsquid.components  import ClassicalFibre,QuantumFibre
from netsquid.pydynaa import Entity,EventHandler,EventType

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


# In[42]:



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
        
        


# In[47]:


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
        print("basis:",basis)
        
        basisList=[basis for i in range(2*self.num_bits)]
        print("basisList:",basisList)
        
        
        #print("mem 1 used?  ",self.processor.get_position_used(2*self.num_bits))
        
        self.myQG_A_measure=QG_A_measure(basisList=basisList,num_bits=2*self.num_bits)
        self.processor.execute_program(
            self.myQG_A_measure,qubit_mapping=[i for  i in range(0, 2*self.num_bits)])
        
        
        self.processor.set_program_done_callback(self.A_getPGoutput,once=True)
        yield self.await_program(processor=self.processor)
        print("self.loc_mesRes",self.loc_mesRes)
        
        self.node.ports[self.portNameC1].tx_output(self.loc_mesRes)
        
        
        port=self.node.ports[self.portNameC2]
        yield self.await_port_input(port)
        Res=port.rx_input().items[0]
        print("A received result:",Res)
        
        

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
        
            
        


# In[53]:


class BobProtocol(NodeProtocol):
    
    def __init__(self,node,processor,num_bits,threshold,
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
        
        print("basisInxList:",self.basisInxList)

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
            print("send rand measurement!")
            self.node.ports[self.portNameC2].tx_output(self.randMeas)
        else:
            print("req error!")
            print(reqMes)
            
        print("waiting for result")
        port = self.node.ports[self.portNameC1]
        yield self.await_port_input(port)
        self.locRes = port.rx_input().items
        print("locRes:",self.locRes)
        
        self.successfulRate=TokenCheck(self.basisInxList,self.randMeas,self.locRes)
        print("B successfulRate:",self.successfulRate)
        
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
        
            
    def B_checkLoss(self,qList):
        num_inx=int(qList[0].name[14:-len('-')-1]) # get index from bits

        self.lossList=[]
        for idx,qubit in enumerate(qList):
            loc_num=int(qubit.name[14:-len('-')-1]) # received qubit
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

    
   


# In[54]:


# implementation & hardware configure
def run_E91_sim(runtimes=1,num_bits=20,fibre_len=10**-9,noise_model=None,
               loss_init=0,loss_len=0):
    
    MyE91List_A=[]  # local protocol list A
    MyE91List_B=[]  # local protocol list B
    
    for i in range(runtimes): 
        
        ns.sim_reset()

        # nodes====================================================================

        nodeA = Node("Alice", port_names=["portQA_1","portCA_1","portCA_2"])
        nodeB = Node("Bob"  , port_names=["portQB_1","portCB_1","portCB_2"])

        # processors===============================================================
        #noise_model=None
        Alice_processor=QuantumProcessor("processor_A", num_positions=3*10**3,
            mem_noise_models=noise_model, phys_instructions=[
            PhysicalInstruction(INSTR_X, duration=20, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_Z, duration=20, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_H, duration=20, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_CNOT,duration=20,q_noise_model=noise_model),
            PhysicalInstruction(INSTR_MEASURE, duration=40, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=40, parallel=True)])


        Bob_processor=QuantumProcessor("processor_B", num_positions=3*10**3,
            mem_noise_models=noise_model, phys_instructions=[
            PhysicalInstruction(INSTR_X, duration=20, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_Z, duration=20, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_H, duration=20, q_noise_model=noise_model),
            PhysicalInstruction(INSTR_CNOT,duration=20,q_noise_model=noise_model),
            PhysicalInstruction(INSTR_MEASURE, duration=40, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=40, parallel=True)])


        # channels==================================================================
        
        MyQChannel=QuantumChannel("QChannel_B->A",delay=10
            ,length=fibre_len
            ,models={"myFibreLossModel": FibreLossModel(p_loss_init=0, p_loss_length=0.25, rng=None)})
        
        
        nodeB.connect_to(nodeA, MyQChannel,
            local_port_name =nodeB.ports["portQB_1"].name,
            remote_port_name=nodeA.ports["portQA_1"].name)
        
        
        MyCChannel= ClassicalChannel("CChannel_A->B",delay=0
            ,length=fibre_len)
        MyCChannel2 = ClassicalChannel("CChannel_B->A",delay=0
            ,length=fibre_len)
        
        
        nodeA.connect_to(nodeB, MyCChannel,
                            local_port_name="portCA_1", remote_port_name="portCB_1")
        nodeB.connect_to(nodeA, MyCChannel2,
                            local_port_name="portCB_2", remote_port_name="portCA_2")
        

        

        Alice_protocol = AliceProtocol(nodeA,Alice_processor,num_bits,waitTime=1)
        Bob_protocol = BobProtocol(nodeB,Bob_processor,num_bits,0.95)
        Bob_protocol.start()
        Alice_protocol.start()
        #ns.logger.setLevel(1)
        stats = ns.sim_run()
        
        
    return MyE91List_A, MyE91List_B







# In[59]:


#test
myErrorModel=T1T2NoiseModel(T1=110, T2=109)
run_E91_sim(1,10,1,noise_model=myErrorModel,loss_init=0,loss_len=0.1) 


# In[18]:


# plot function
import matplotlib.pyplot as plt

def E91_plot():
    y_axis=[]
    x_axis=[]
    run_times=20
    num_bits=50
    min_dis=1
    max_dis=10
    
    myErrorModel=DepolarNoiseModel(depolar_rate=1000)

    # first curve
    for i in range(min_dis,max_dis):
        key_sum=0.0
        x_axis.append(i/10)
        key_list_A,key_list_B=run_E91_sim(run_times,num_bits,i/10
            ,noise_model=myErrorModel,loss_init=0,loss_len=0.1) 
        #feed runtimes, numberof bits and distance, use default loss model
        #print("key_list_A: ",key_list_A)
        for keyA,keyB in zip(key_list_A,key_list_B):
            if keyA==keyB and keyA != None:  
                #print("len keyA:",len(keyA))
                key_sum+=len(keyA)
        y_axis.append(key_sum/run_times/num_bits)
        
    plt.plot(x_axis, y_axis, 'go-',label='loss_len=0.2')
    
    
    y_axis.clear() 
    x_axis.clear()
    
    
    # second curve
    for i in range(min_dis,max_dis):
        key_sum=0.0
        x_axis.append(i/10)
        key_list_A,key_list_B=run_E91_sim(run_times,num_bits,i/10
            ,noise_model=myErrorModel,loss_init=0,loss_len=0.9) 
        
        for keyA,keyB in zip(key_list_A,key_list_B):
            if keyA==keyB and keyA != None: 
                key_sum+=len(keyA)
        y_axis.append(key_sum/run_times/num_bits)
        
    plt.plot(x_axis, y_axis, 'bo-',label='loss_len=0.4')
    
    
    
    plt.ylabel('average key length/max qubits length')
    plt.xlabel('fibre lenth (km)')
    
    
    plt.legend()
    plt.savefig('plot.png')
    plt.show()

    

E91_plot()


# In[ ]:




