#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.protocols import NodeProtocol
from netsquid.qubits.operators import X,H,Z
from netsquid.components  import ClassicalFibre,QuantumFibre

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
        print("Comparing error! length of basis does not match!")
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



'''
Key length filter
'''
# function that filters vowels 
def lenfilter(var): 
    if len(var) <= 10 and len(var) > 6: 
        return True
    else: 
        return False
  


# In[3]:


# class of quantum program
class QG_A_qPrepare(QuantumProgram):
    
    def __init__(self,num_bits=1):
        self.num_bits=num_bits
        super().__init__()
        
    def program(self):
        qList_idx=self.get_qubit_indices(2*self.num_bits)
        # create multiEPR
        #print("A QG_A_qPrepare")
        for i in range(2*self.num_bits):
            #self.apply(INSTR_INIT, qList_idx[i])
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
    
    def __init__(self,node,processor,num_bits,
                port_names=["portQA_1","portCA_1","portCA_2"]):
        super().__init__()
        self.num_bits=num_bits
        self.node=node
        self.processor=processor
        self.portNameQ1=port_names[0]
        self.portNameC1=port_names[1]
        self.portNameC2=port_names[2]
        self.EPRList=None
        self.basisList=Random_basis_gen(self.num_bits)
        self.loc_mesRes=[]
        self.key=None
        self.sourceQList=[]
        
        #generat qubits from source
        self.A_Source = QSource("Alice_source"
            ,status=SourceStatus.EXTERNAL) # enable frequency
        self.A_Source.ports["qout0"].bind_output_handler(self.storeSourceOutput)
        
    # =======================================A run ============================
    def run(self):

        
        # A generat qubits
        self.A_genQubits(self.num_bits,1e9)
        
        # wait
        yield self.await_program(processor=self.processor)
        
        
        #yield self.await_program(processor=self.processor)
        # send qubits
        self.A_sendEPR()
        
        # receive B basis
        port=self.node.ports[self.portNameC1]
        yield self.await_port_input(port)
        basis_B = port.rx_input().items
        
        
        #self.A_measure()
        self.myQG_A_measure=QG_A_measure(
            basisList=self.basisList,num_bits=self.num_bits)
        self.processor.execute_program(
            self.myQG_A_measure,qubit_mapping=[i for  i in range(0, 2*self.num_bits)])
        
        
        # get A meas
        self.processor.set_program_done_callback(self.A_getPGoutput,once=True)
        
        # send A basis to B
        self.node.ports[self.portNameC2].tx_output(self.basisList)
        
        
        # compare basis
        yield self.await_program(processor=self.processor)
        
        
        self.loc_mesRes=Compare_basis(self.basisList,basis_B,self.loc_mesRes)
        
        self.key=''.join(map(str, self.loc_mesRes))
        #print("A key:",self.key)

    def storeSourceOutput(self,qubit):
        self.sourceQList.append(qubit.items[0])
        if len(self.sourceQList)==2*self.num_bits:
            self.processor.put(qubits=self.sourceQList)
            
            #self.A_sendEPR()
            # apply H detector
            PG_qPrepare=QG_A_qPrepare(num_bits=self.num_bits)
            self.processor.execute_program(
                PG_qPrepare,qubit_mapping=[i for  i in range(0, 2*self.num_bits)])


    def A_genQubits(self,num,freq=1e9):
        
        
        #set clock
        clock = Clock("clock", frequency=freq, max_ticks=2*num)
        try:
            clock.ports["cout"].connect(self.A_Source.ports["trigger"])
        except:
            pass
            #print("alread connected") 
            
        clock.start()
        
            
    def A_sendEPR(self):
        #print("A_sendEPR")
        inx=list(range(1,2*self.num_bits+1,2))
        payload=self.processor.pop(inx)
        self.node.ports[self.portNameQ1].tx_output(payload)
        

    def A_getPGoutput(self):
        self.loc_mesRes=getPGoutput(self.myQG_A_measure)


# In[5]:


class BobProtocol(NodeProtocol):
    
    def __init__(self,node,processor,num_bits,
                port_names=["portQB_1","portCB_1","portCB_2"]):
        super().__init__()
        self.num_bits=num_bits
        self.node=node
        self.processor=processor
        self.qList=None
        self.loc_measRes=[-1]*self.num_bits
        self.portNameQ1=port_names[0]
        self.portNameC1=port_names[1]
        self.portNameC2=port_names[2]
        # init value assume that all qubits are lost
        self.key=None
        self.PG_B=None
        self.lossList=[]
        self.firstLoss=0
        self.endTime=None

    # =======================================B run ============================
    def run(self):
        
        qubitList=[]
        
        #receive qubits from A
        
        port = self.node.ports[self.portNameQ1]
        qubitList=[]
        
        #receive qubits from A
        
        
        yield self.await_port_input(port)
        qubitList.append(port.rx_input().items)
        #print("B received qubits:",qubitList)
        self.B_checkLoss(qubitList[0])
        
        
        #put qubits into B memory
        for qubit in qubitList:
            self.processor.put(qubit)
        
        self.myQG_B_measure=QG_B_measure(
            basisList=self.basisList,num_bits=self.num_bits)
        self.processor.execute_program(
            self.myQG_B_measure,qubit_mapping=[i for  i in range(0,self.num_bits)])
        
        # get meas result
        self.processor.set_program_done_callback(self.B_getPGoutput,once=True)
        
        yield self.await_program(processor=self.processor)
        

        # add Loss case
        self.loc_measRes=AddLossCase(self.lossList,self.loc_measRes)
        self.basisList=AddLossCase(self.lossList,self.basisList)
        
        # add first loss
        if self.firstLoss>=1:   
            for i in range(self.firstLoss):
                self.loc_measRes.insert(0,-1)
                self.basisList.insert(0,-1)
        
        # self.B_send_basis()
        self.node.ports[self.portNameC1].tx_output(self.basisList)
        
        # wait for A's basisList
        port=self.node.ports[self.portNameC2]
        yield self.await_port_input(port)
        basis_A=port.rx_input().items
        #print("B received basis_A:",basis_A)
        
        self.loc_measRes=Compare_basis(self.basisList,basis_A,self.loc_measRes)
        
        self.key=''.join(map(str, self.loc_measRes))
        #print("B key:",self.key)
        self.endTime=ns.sim_time()
        
        
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

    
    def B_getPGoutput(self):
        self.loc_measRes=getPGoutput(self.myQG_B_measure)


# In[6]:


# implementation & hardware configure
#import difflib
from difflib import SequenceMatcher

def run_E91_sim(runtimes=1,num_bits=20,fibre_len=10**-9,memNoiseMmodel=None,processorNoiseModel=None
               ,loss_init=0,loss_len=0):
    
    MyE91List_A=[]  # local protocol list A
    MyE91List_B=[]  # local protocol list B
    MyKeyRateList=[]
    
    for i in range(runtimes): 
        
        ns.sim_reset()

        # nodes====================================================================

        nodeA = Node("Alice", port_names=["portQA_1","portCA_1","portCA_2"])
        nodeB = Node("Bob"  , port_names=["portQB_1","portCB_1","portCB_2"])

        # processors===============================================================
        #noise_model=None
        Alice_processor=QuantumProcessor("processor_A", num_positions=3*10**3,
            mem_noise_models=memNoiseMmodel, phys_instructions=[
            PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
            PhysicalInstruction(INSTR_X, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_H, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_CNOT,duration=10,q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_MEASURE, duration=10,q_noise_model=processorNoiseModel, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=10,q_noise_model=processorNoiseModel, parallel=True)])


        Bob_processor=QuantumProcessor("processor_B", num_positions=3*10**3,
            mem_noise_models=memNoiseMmodel, phys_instructions=[
            PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
            PhysicalInstruction(INSTR_X, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_H, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_CNOT,duration=1,q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_MEASURE, duration=10,q_noise_model=processorNoiseModel, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=10,q_noise_model=processorNoiseModel, parallel=True)])


        # channels==================================================================
        
        MyQChannel=QuantumChannel("QChannel_A->B",delay=10
            ,length=fibre_len
            ,models={"myFibreLossModel": FibreLossModel(p_loss_init=0.2, p_loss_length=0.25, rng=None)})
        
        
        nodeA.connect_to(nodeB, MyQChannel,
            local_port_name =nodeA.ports["portQA_1"].name,
            remote_port_name=nodeB.ports["portQB_1"].name)
        

        MyCChannel = ClassicalChannel("CChannel_B->A",delay=0
            ,length=fibre_len)
        MyCChannel2= ClassicalChannel("CChannel_A->B",delay=0
            ,length=fibre_len)
        

        nodeB.connect_to(nodeA, MyCChannel,
                            local_port_name="portCB_1", remote_port_name="portCA_1")
        nodeA.connect_to(nodeB, MyCChannel2,
                            local_port_name="portCA_2", remote_port_name="portCB_2")

        
        startTime=ns.sim_time()
        #print("startTime:",startTime)
        
        Alice_protocol = AliceProtocol(nodeA,Alice_processor,num_bits)
        Bob_protocol = BobProtocol(nodeB,Bob_processor,num_bits)
        Bob_protocol.start()
        Alice_protocol.start()
        #ns.logger.setLevel(1)
        stats = ns.sim_run()
        
        endTime=Bob_protocol.endTime
        #print("endTime:",endTime)
        
        MyE91List_A.append(Alice_protocol.key)
        MyE91List_B.append(Bob_protocol.key)
        
        
        #simple key length calibration
        s = SequenceMatcher(None, Alice_protocol.key, Bob_protocol.key)# unmatched rate
        MyKeyRateList.append((len(Bob_protocol.key)*(1-s.ratio()))*10**9/(endTime-startTime)) #second
        
    return MyE91List_A, MyE91List_B, MyKeyRateList







# In[7]:


# plot function
import matplotlib.pyplot as plt

def E91_plot():
    y_axis=[]
    x_axis=[]
    run_times=10
    num_bits=50
    min_dis=0
    max_dis=80
    
    mymemNoiseMmodel=T1T2NoiseModel(T1=11, T2=10)
    myprocessorNoiseModel=DepolarNoiseModel(depolar_rate=200)

    # first curve
    for i in range(min_dis,max_dis,5):
        
        x_axis.append(i)
        key_list_A,key_list_B,keyRateList=run_E91_sim(run_times,num_bits,fibre_len=i
            ,processorNoiseModel=myprocessorNoiseModel,memNoiseMmodel=mymemNoiseMmodel) 
        
        y_axis.append(sum(keyRateList)/run_times/10**6)
        
        
        
    plt.plot(x_axis, y_axis, 'go-',label='depolar_rate=200Hz')
    
    '''
    y_axis.clear() 
    x_axis.clear()
    
    
    myprocessorNoiseModel=DepolarNoiseModel(depolar_rate=2000)
    # second curve
    for i in range(min_dis,max_dis,5):
        
        x_axis.append(i)
        key_list_A,key_list_B,keyRateList=run_E91_sim(run_times,num_bits,fibre_len=i
            ,processorNoiseModel=myprocessorNoiseModel,memNoiseMmodel=mymemNoiseMmodel) 
        
        y_axis.append(sum(keyRateList)/run_times)
        
        
        
    plt.plot(x_axis, y_axis, 'bo-',label='depolar_rate=2000')
    '''
    
    plt.title('QKD E91')
    plt.ylabel('key rate Mb/s')
    plt.xlabel('fibre lenth (km)')
    
    
    plt.legend()
    plt.savefig('keyRate8.png')
    plt.show()

    

E91_plot()


# In[ ]:









#test
mymemNoiseMmodel=T1T2NoiseModel(T1=11, T2=10)
myprocessorNoiseModel=DepolarNoiseModel(depolar_rate=500)

toWrite=run_E91_sim(runtimes=1,num_bits=20,fibre_len=20
        ,memNoiseMmodel=mymemNoiseMmodel,processorNoiseModel=myprocessorNoiseModel) #10**-9
print(toWrite)
s1=toWrite[0]
s2=toWrite[1] 
print(s1)
print(s2)


# In[8]:


# key pairs generation

#myErrorModel=DepolarNoiseModel(depolar_rate=50000)
myErrorModel=T1T2NoiseModel(T1=1100, T2=1000)
toWrite=run_E91_sim(runtimes=1,num_bits=10**4,fibre_len=10,noise_model=myErrorModel) #10**-9
#print(toWrite)

listToPrint=''
listToPrint=str(toWrite)
print(listToPrint)
outF = open("keyOutput8.txt", "w")
outF.writelines(listToPrint)
outF.close()


