#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from random import randint

from netsquid.qubits.qformalism import *
from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.components.models.qerrormodels import *


# In[8]:


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
    '''
    print("loc_basis_list:",loc_basis_list)
    print("rem_basis_list:",rem_basis_list)
    print("len loc_basis_list:",len(loc_basis_list))
    print("len rem_basis_list:",len(rem_basis_list))
    '''
    
    if len(loc_basis_list) != len(rem_basis_list): #should be  len(num_bits)
        #print("Comparing error! length of basis does not match! Check classical port")
        return -1
    
    popList=[]
    
    for i in range(len(rem_basis_list)):
        if loc_basis_list[i] != rem_basis_list[i]:
            popList.append(i)
    
    #print("pop list:",popList)
    
    for i in reversed(popList): 
        if loc_res:
            loc_res.pop(i)
        
    return loc_res


'''
Ramdomly measure a list of qubits and keep the basis used. If there's a qubit missing, the list append -1.
P.S. The qubit index goes 2 by 2 in this function. Doesn't work if the first qubit of qList is lost.

input:
    num_bits: Number of qubits.(int)
    qlist: A qubits list to measure.(qubit)
output:
    basisList: A list of basis used on measurement. (list of int)
    loc_res_measure: A list of measurement result. (list of int)
'''


def Random_ZX_measure(num_bits,qlist):
    num_start=int(qlist[0].name[3:-len('-')-1])# get value after qubit name "QS#<num>-x"
    basisList = []*num_bits  # set boundary
    loc_res_measure=[]*num_bits  # set boundary
    ind=0
    for i in range(num_start,num_start+2*num_bits,2):
        if ind <= len(qlist)-1 and qlist[ind]!=None:
            if int(qlist[ind].name[3:-len('-')-1]) == i:
                rbit = randint(0,1) # 0:Z 1:X
                if rbit:
                    basisList.append(1)  #'X'
                    loc_res_measure.append(ns.qubits.qubitapi.
                        measure(qlist[ind],observable=X)[0]) #measure in Hadamard basis
                else:
                    basisList.append(0)  #'Z'
                    loc_res_measure.append(ns.qubits.qubitapi.
                        measure(qlist[ind],observable=Z)[0]) #measure in standard basis
                ind+=1
            else:
                basisList.append(-1)
                loc_res_measure.append(-1)
        else:
            basisList.append(-1)
            loc_res_measure.append(-1)
    return basisList,loc_res_measure



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
    
    newDict=sorted({int(k) : v for k, v in tempDict.items()}.items())
    
    #take value
    for k, v in newDict:
        resList.append(v[0])
    
    #print("getPGoutput output List:",resList)
                   
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


# In[9]:


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


# apply measurement on qubits in quantum memory according to given basis. 
# sometimes fail
class QG_B_measure(QuantumProgram):
    
    def __init__(self,basisList,num_bits,firstLoss):
        self.basisList=basisList
        self.num_bits=num_bits
        self.firstLoss=firstLoss
        super().__init__()


    def program(self):   
        #if self.firstLoss==False:
        for i in range(len(self.basisList)):
            if self.basisList[i] == 0:                                # standard basis
                self.apply(INSTR_MEASURE, qubit_indices=i, output_key=str(i),physical=False) #True
            else:                              # 1 case               # Hadamard basis
                self.apply(INSTR_MEASURE_X, qubit_indices=i, output_key=str(i),physical=False)       
        yield self.run(parallel=False)


# In[10]:


class E91_A(LocalProtocol):
    
    def __init__(self,node,processor,num_bits
        ,portNameQ1="portQA_1",portNameC1="portCA_1",portNameC2="portCA_2"):
        self.node=node
        self.processor=processor
        self.EPRList=None
        self.num_bits=num_bits
        self.Basis=Random_basis_gen(self.num_bits)
        self.loc_mesRes=[]
        self.key=None
        self.portNameQ1=portNameQ1
        self.portNameC2=portNameC2
        
        self.node.ports[portNameC1].bind_input_handler(self.A_compare_basis)


    def A_prepare_qubits(self):
        basisList=Random_basis_gen(self.num_bits)
        
        # Do Qprogram
        PG_A=QG_A_qGen(num_bits=self.num_bits)
        self.processor.execute_program(PG_A,qubit_mapping=[i for  i in range(0, 2*self.num_bits)])
        self.processor.set_program_done_callback(self.A_sendEPR,once=True)
        
        #debug
        #print("A after qPG: used position",self.processor.num_used_positions)
        yield self.run()
    
    
    
    def A_sendEPR(self):
        self.processor.sendfromMem(senderNode=self.node
            ,inx=list(range(1,2*self.num_bits+1,2))   # get even index in qList
            ,senderPortName=self.node.ports[self.portNameQ1].name)

        
    def A_measure(self):
        qList=[]
        qList=self.processor.pop(positions=list(range(0,2*self.num_bits,2)))
        for i in range(len(qList)):
            if(self.Basis[i]==0):
                self.loc_mesRes.append(ns.qubits.qubitapi
                    .measure(qList[i],observable=Z)[0]) #measure in standard basis
            else:
                self.loc_mesRes.append(ns.qubits.qubitapi
                    .measure(qList[i],observable=X)[0]) #measure in Hadamard basis
        '''
        #debug
        #print("\nA Basis:",self.Basis)
        #print("A mesRes:",self.loc_mesRes)
        '''
        
    def A_compare_basis(self,basis_B):
        # A measure
        self.A_measure()
        
        # A compare
        if len(self.loc_mesRes)>0 and self.loc_mesRes != -1:
            '''
            print("A basis:",self.Basis)
            print("A basis len:",len(self.Basis))
            print("A basis_B.items:",basis_B.items)
            print("A basis_B.items len:",len(basis_B.items))
            print("A loc_mesRes before compare:",self.loc_mesRes)
            '''
            
            self.loc_mesRes=Compare_basis(self.Basis,basis_B.items,self.loc_mesRes)
            #debug 
            '''
            print("\nA self.loc_mesRes:",self.loc_mesRes)
            print("A len self.loc_mesRes:",len(self.loc_mesRes))
            print("\n")
            '''
            self.key=''.join(map(str, self.loc_mesRes))
            self.node.ports[self.portNameC2].tx_output(self.Basis)
        else:
            print("Error!! A no loc_mesRes")
        
        


# In[11]:


class E91_B(LocalProtocol):
    
    def __init__(self,node,processor,num_bits
        ,portNameQ1="portQB_1",portNameC1="portCB_1",portNameC2="portCB_2"):
        self.node=node
        self.processor=processor
        self.qList=None
        self.num_bits=num_bits
        self.Basis=[] 
        self.loc_mesRes=[-1]*self.num_bits   # init value assume that all qubits are lost
        self.key=None
        self.portNameC1=portNameC1
        self.PG_B=None
        self.lossList=[]
        self.firstLoss=0
        
        self.node.ports[portNameQ1].bind_input_handler(self.B_receive)
        self.node.ports[portNameC2].bind_input_handler(self.B_compare_basis)
    
        
    def B_receive(self,message):
        qList=message.items
        for i in self.B_measure(qList):
            pass
        
        
        
    def B_measure(self,qList):
        #print("Bob received qubits:",qList)
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

        # make basis
        self.Basis=Random_basis_gen(len(qList)) # exact length needed
        
        # check for first N qubit loss
        if self.num_bits-len(self.lossList)>len(qList):
            self.firstLoss=self.num_bits-len(qList)-len(self.lossList)  #first qubit loss detected
        else:
            self.firstLoss=0
        
        #print("self.firstloss:",self.firstLoss)
        
        # put qubits in B processor(Qmem)
        self.processor.put(qubits=qList)

        #do QG
        self.PG_B=QG_B_measure(basisList=self.Basis,num_bits=self.num_bits,firstLoss=self.firstLoss)

        self.processor.execute_program(self.PG_B,qubit_mapping=[i for  i in range(0,self.num_bits)])
        self.processor.set_program_done_callback(self.B_send_basis,once=True)
        self.processor.set_program_fail_callback(self.B_failCase,once=True)
        
        yield self.run()
        
        
    def B_send_basis(self):
        self.loc_mesRes=getPGoutput(self.PG_B)
        '''
        print("before add loss:")
        print("loc_mesRes:",self.loc_mesRes)
        print("self.Basis:",self.Basis)
        print("len loc_mesRes:",len(self.loc_mesRes))
        print("len self.Basis:",len(self.Basis))
        '''
        self.loc_mesRes=AddLossCase(self.lossList,self.loc_mesRes)
        self.Basis=AddLossCase(self.lossList,self.Basis)
        '''
        print("after add loss:")
        print("loc_mesRes:",self.loc_mesRes)
        print("self.Basis:",self.Basis)
        print("len loc_mesRes:",len(self.loc_mesRes))
        print("len self.Basis:",len(self.Basis))
        '''
        
        if self.firstLoss>=1:   
            for i in range(self.firstLoss):
                self.loc_mesRes.insert(0,-1)
                self.Basis.insert(0,-1)
        '''        
        print("after add first loss:")
        print("loc_mesRes:",self.loc_mesRes)
        print("self.Basis:",self.Basis)
        print("len loc_mesRes:",len(self.loc_mesRes))
        print("len self.Basis:",len(self.Basis))
        '''
        #send
        self.node.ports[self.portNameC1].tx_output(self.Basis)
        
        
    def B_compare_basis(self,basis_A):
        if len(self.loc_mesRes)>1: # exclude empty case
            #debug           
            '''
            print("B basis:",self.Basis)
            print("B basis len:",len(self.Basis))
            print("B basis_A.items:",basis_A.items)
            print("B basis_A.items len:",len(basis_A.items))
            print("B loc_mesRes before compare:",self.loc_mesRes)
            '''
            self.loc_mesRes=Compare_basis(self.Basis,basis_A.items,self.loc_mesRes)
            
            
            '''
            print("\nB self.loc_mesRes:",self.loc_mesRes)
            print("B len self.loc_mesRes:",len(self.loc_mesRes))
            print("\n")
            '''
            
            self.key=''.join(map(str, self.loc_mesRes))
            
        else:
            print("Error! No measurement results on B!")
        
    def B_failCase(self):
        print("Error on B measurement in quantum program")
        


# In[20]:


# implementation & hardware configure

def run_E91_sim(runtimes=1,num_bits=20,fibre_len=10**-9,noise_model=None):
    
    MyE91List_A=[]  # local protocol list A
    MyE91List_B=[]  # local protocol list B
    
    for i in range(runtimes): 
        
        ns.sim_reset()
        
        # Hardware configuration
        Alice_processor=sendableQProcessor("processor_A", num_positions=100,
                    mem_noise_models=None, phys_instructions=[
                    PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
                    PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_CNOT,duration=1,q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True)],
                    topologies=[None, None, None, None, None, None])

        #mem_noise_models=[DepolarNoiseModel(0)] * 100
        Bob_processor=sendableQProcessor("processor_B", num_positions=100,
                    mem_noise_models=None, phys_instructions=[
                    PhysicalInstruction(INSTR_INIT, duration=1, parallel=True),
                    PhysicalInstruction(INSTR_X, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_H, duration=1, q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_CNOT,duration=1,q_noise_model=noise_model),
                    PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True)],
                    topologies=[None, None, None, None, None, None])


        node_A = Node("A",ID=0,port_names=["portQA_1","portCA_1","portCA_2"])
        node_B = Node("B",ID=1,port_names=["portQB_1","portCB_1","portCB_2"])


        # connection
        MyQfiber=QuantumFibre("QFibre_A->B", length=fibre_len,quantum_loss_model=None) 
        # default value:
        # p_loss_init=0.2, p_loss_length=0.25,depolar_rate=0, c=200000, models=None


        MyCfiber=DirectConnection("CFibreConn_B->A",
            ClassicalFibre("CFibre_B->A", length=fibre_len))


        MyCfiber2=DirectConnection("CFibreConn_A->B",
            ClassicalFibre("CFibre_A->B", length=fibre_len))


        node_A.connect_to(node_B, MyQfiber,
            local_port_name =node_A.ports["portQA_1"].name,
            remote_port_name=node_B.ports["portQB_1"].name)


        node_B.connect_to(node_A, MyCfiber,
                    local_port_name="portCB_1", remote_port_name="portCA_1")

        node_A.connect_to(node_B, MyCfiber2,
                    local_port_name="portCA_2", remote_port_name="portCB_2")
        
        
        
        Alice=E91_A(node=node_A,processor=Alice_processor,num_bits=num_bits
            ,portNameQ1="portQA_1",portNameC1="portCA_1",portNameC2="portCA_2")
        Bob=E91_B(node=node_B,processor=Bob_processor,num_bits=num_bits
            ,portNameQ1="portQB_1",portNameC1="portCB_1",portNameC2="portCB_2")
        
        
        for i in Alice.A_prepare_qubits():
            pass
        
        ns.sim_run()
        
        MyE91List_A.append(Alice.key)
        MyE91List_B.append(Bob.key)
        
    return MyE91List_A, MyE91List_B


#unit test
#run_E91_sim(runtimes=1,num_bits=20,fibre_len=10**-9,noise_model=None)



# In[7]:


# plot function
import matplotlib.pyplot as plt

def E91_plot():
    y_axis=[]
    x_axis=[]
    run_times=10
    num_bits=20
    min_dis=1000
    max_dis=15000

    # first curve
    for i in range(min_dis,max_dis,1000):
        key_sum=0.0
        x_axis.append(1.*i/1000)
        key_list_A,key_list_B=run_E91_sim(run_times,num_bits,1.*i/1000
            ,noise_model=None) 
        #feed runtimes, numberof bits and distance, use default loss model
        #print("key_list_A: ",key_list_A)
        for keyA,keyB in zip(key_list_A,key_list_B):
            if keyA==keyB and keyA != None:  #else error happend, drop key, count 0 length
                #print("len keyA:",len(keyA))
                key_sum+=len(keyA)
        y_axis.append(key_sum/run_times/num_bits)
        
    plt.plot(x_axis, y_axis, 'go-',label='without noise model')
    
    
    y_axis.clear() 
    x_axis.clear()
    
    # second curve
    for i in range(min_dis,max_dis,1000):
        key_sum=0.0
        x_axis.append(1.*i/1000)
        key_list_A,key_list_B=run_E91_sim(run_times,num_bits,1.*i/1000
            ,noise_model=DepolarNoiseModel(depolar_rate=500)) 
        #feed runtimes, numberof bits and distance, use default loss model
        for keyA,keyB in zip(key_list_A,key_list_B):
            if keyA==keyB and keyA != None:  #else error happend, drop key, count 0 length
                key_sum=key_sum+len(keyA)
        y_axis.append(key_sum/run_times/num_bits)
        
    plt.plot(x_axis, y_axis, 'bo-',label='DepolarNoiseModel depolar rate:500')
        
    plt.ylabel('average key length/original qubits length')
    plt.xlabel('fibre lenth (km)')
    
    
    plt.legend()
    plt.savefig('plot.png')
    plt.show()

    

E91_plot()

