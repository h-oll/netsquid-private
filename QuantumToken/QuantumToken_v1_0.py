#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.protocols import Protocol
from netsquid.qubits import create_qubits
from netsquid.qubits.operators import *
from netsquid.nodes.connections import DirectConnection
from netsquid.components import ClassicalFibre,QuantumFibre,FibreLossModel,QuantumMemory, DepolarNoiseModel,DephaseNoiseModel
from netsquid.components.models import FixedDelayModel, GaussianDelayModel, FibreDelayModel

from random import seed, randint
import time


# In[7]:



def random_basis_gen(num_bits):
    seed(randint(0, 2**num_bits))
    opList=[]
        
    for _ in range(num_bits):
        rbit = randint(0,1)
        if rbit==0:
            opList.append(0) 
        elif rbit==1:
            opList.append(1)
        else:
            print("randint ERROR!!\n")
            return 0
    print("R opList=",opList)
    return opList

    
# returns a list of stats and a list of qubits
def create_random_qubits(num_bits):
    seed(randint(0, 4**num_bits))
    res_state=[]
    qlist=[]
    qlist=create_qubits(num_bits,system_name="Q") 
    for i in range(0,num_bits):
        res_state.append(randint(0,3))
    for a,b in zip(res_state, qlist):
        if   a == 0: # 0 state
            #print("0",b.qstate.dm)
            pass
        elif a == 1: # 1 state    #X
            X | b
            #print("1",b.qstate.dm)
        elif a == 2: # + state    #H
            H | b
            #print("+",b.qstate.dm)
        elif a == 3: # - state    #XH
            X | b
            H | b
            #print("-",b.qstate.dm)
        else :
            print("Create random bits ERROR!!")
    return res_state, qlist



def measure_by_basis(basisList,qList):
    if len(basisList)<len(qList): 
        print("Quantum list is too long! ERROR!!")
        return 0
    else:
        res_measurement=[0]*len(basisList) #init to 0
        
        for q in qList:
            a=int(q.name[1:]) #get qubit index
            if basisList[a]==0:
                res_measurement[a]=ns.qubits.qubitapi.measure(q,observable=Z) #measure in standard basis
            elif basisList[a]==1:
                res_measurement[a]=ns.qubits.qubitapi.measure(q,observable=X) #measure in Hadamard basis
            else:
                print("measuring ERROR!!\n")    
        return res_measurement


# not done yet
def Match_rate_Cal(num_bits,challenge,stateList,res_measure,threshold):
    
    False_count=0
    
    for a,b,c in zip(challenge,stateList,res_measure):
        #print(a,b,c)
            
        if int(a)==0 :
            if b<=1 and c[1]==1 and b!=c[0]:
                False_count+=1
            elif b>2 and c[1]==1:
                False_count+=1
            else:
                pass
        elif int(a)==1:
            if b<=1 and c[1]==1:
                False_count+=1
            elif b>2 and c[1]==1 and b!=c[0]:
                False_count+=1
            else:
                pass
        else:
            print("ERROR in challenge value!!")
        

    return 1.0-False_count/num_bits
    
    


# In[11]:


class QuantumToken(Protocol):
    
    
    # QuantumToken functions ==============================================================================
    
    def B_prepare_send(self, num_bits):
        
        self.stateList, tokenQlist=create_random_qubits(num_bits)
        # send qubits to A (tickets)
        print("B_sending qubits with state:",self.stateList)
        self.node_B.ports["portQ_B"].tx_output(tokenQlist)
        #self.portQ_B.tx_output(tokenQlist)

        
    def A_ask_challenge(self, qList):
        print("sleeping...")
        time.sleep(5)
        qList=qList.items
        print("A_ask_challenge: ")
        self.tokenQlist=qList
        message="10101"
        self.node_A.ports["portC_A1"].tx_output(message)
        #self.portC_A.tx_output(message)
        
        
    def B_send_challenge(self,message):
        print("B_send_challenge: ")
        message=message.items[0]
        if str(message)=="10101":
            
            self.challenge=random_basis_gen(self.num_bits)
            print("sending challenge:", self.challenge)
            self.node_B.ports["portC_B2"].tx_output(self.challenge)
        else:
            pass
        
        
    def A_measure_send(self, basisList):
        print("A_measure_send: ")         # pause, save qList in quantum memory
        res_measure = measure_by_basis(basisList.items,self.tokenQlist)
        #print("res_measure: ",res_measure)
        self.node_A.ports["portC_A3"].tx_output(res_measure)
        
        
    def B_evaluate_reply(self,res_measure):
        res_measure=res_measure.items
        print("B_evaluate_reply")
        print("res_measure: ",res_measure)
        print("self.stateList: ",self.stateList)
        print("self.challenge: ",self.challenge)
        
        success_rate=Match_rate_Cal(self.num_bits,self.challenge,self.stateList,res_measure,self.validation_threshold)
        
        
        print("success_rate: ",success_rate)
        if self.validation_threshold<=success_rate:
            print("Accepted!")
            self.permission=True
        else:
            print("Aborted!")
            self.permission=False
    
    
    # basic functions ========================================================================
    def __init__(self, num_bits=8,fiberLenth=1): 
        super().__init__()
        self.node_A = Node("A",ID=0,port_names=["portQ_A","portC_A1","portC_A2","portC_A3","portC_A4"])
        self.node_B = Node("B",ID=1,port_names=["portQ_B","portC_B1","portC_B2","portC_B3","portC_B4"])

        self.MyQfiber = None
        #self.MyCfiber = []
        self.fiberLenth = fiberLenth
        self.num_bits = num_bits
        self.stateList = None
        self.tokenQlist = None
        self.challenge = []   #B to A
        self.validation_threshold = 0.95
        self.permission = False
        self.start()
        
    def stop(self):
        super().stop()
        self._running = False
        
        
    def is_connected(self):
        super().is_connected()
        pass
        
        
    def start(self):
        super().start()
        
        # connect and connect quantum fibres
        self.MyQfiber=QuantumFibre("QFibre_AB", length=self.fiberLenth, 
            loss_model=None, #FibreLossModel(p_loss_length=self.fibre_loss_length,p_loss_init=self.fibre_loss_init), depolar_rate=0, 
            noise_model="default") 
        
        # one directional
        self.node_B.connect_to(self.node_A, self.MyQfiber,
            local_port_name="portQ_B", remote_port_name="portQ_A")

        
        # create classical fibre
        for i in range(1,4):
            MyCfiber=DirectConnection("CFibreConn_"+str(i),
                ClassicalFibre("CFibre_A->B"+str(i), length=self.fiberLenth),
                ClassicalFibre("CFibre_B->A"+str(i), length=self.fiberLenth))
            if i%2 == 1 :
                self.node_A.connect_to(self.node_B, MyCfiber,
                    local_port_name="portC_A"+str(i), remote_port_name="portC_B"+str(i))
            else:
                self.node_B.connect_to(self.node_A, MyCfiber,
                    local_port_name="portC_B"+str(i), remote_port_name="portC_A"+str(i))
        
        
        
        
        
        #self.node_A.connect_to(self.node_B, self.MyCfiber,
            #local_port_name="portC_A", remote_port_name="portC_B")
        
        
        # quantum memory========================================================
        A_memory = QuantumMemory("QMemory_A", self.num_bits,ID=0, 
            memory_noise_models=DepolarNoiseModel(depolar_rate=0.5,time_independent=True))
        #QuantumNoiseModel
        #DephaseNoiseModel(dephase_rate
        #DepolarNoiseModel(depolar_rate
        self.node_A.add_subcomponent(A_memory)
        
        B_memory =QuantumMemory("QMemory_B", self.num_bits,ID=1, 
            memory_noise_models=DepolarNoiseModel(depolar_rate=0.5))
        self.node_B.add_subcomponent(B_memory)
        

        #set callback functions===================================================
        self.node_A.ports["portQ_A"].bind_input_handler(self.A_ask_challenge)
        #self.portC_B.bind_input_handler(self.B_send_challenge)
        self.node_B.ports["portC_B1"].bind_input_handler(self.B_send_challenge)
        self.node_A.ports["portC_A2"].bind_input_handler(self.A_measure_send)
        self.node_B.ports["portC_B3"].bind_input_handler(self.B_evaluate_reply)
        
        
        
        self.B_prepare_send(self.num_bits)
        
            



# In[12]:


def run_QuantumToken_sim(run_times=1):
    ns.sim_reset()
    
    QuantumTokenList=[]
    for i in range(run_times): 
        
        print("The ",i,"th run...")
        QuantumTokenList.append(QuantumToken(num_bits=30).stateList) #give an attribute
        
    ns.sim_run()
    #ns.logger.setLevel(1)
    
    return QuantumTokenList


# In[13]:


run_QuantumToken_sim(1)


# In[ ]:




