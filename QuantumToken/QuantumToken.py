#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.protocols import Protocol
from netsquid.qubits import create_qubits
from netsquid.qubits.operators import *
from netsquid.nodes.connections import DirectConnection
from netsquid.components import ClassicalFibre,QuantumFibre,FibreLossModel,QuantumMemory, DepolarNoiseModel,DephaseNoiseModel
from netsquid.components.models import FixedDelayModel, GaussianDelayModel, FibreDelayModel
from netsquid.pydynaa import Entity,EventHandler,EventType

from random import seed, randint
import time


# In[13]:


'''
Genenrate a list with elements 0 or 1 randomly.
Indicates basis later on.
input:
    Length of list.
output:
    A list consists of 0 and 1.
'''
def Random_basis_gen(length):
    return [randint(0,1) for i in range(length)]


'''
Randomly produce qubits in four states, and record the states.

input:
    Number of qubits in the list.

output:
    A list of stats and a list of qubits
'''   
def Create_random_qubits(num_bits):
    res_state=[]
    qlist=[]
    qlist=create_qubits(num_bits,system_name="Q") 
    for i in range(0,num_bits):
        res_state.append(randint(0,3))
    for a,b in zip(res_state, qlist):
        if   a == 0: # 0 state
            pass
        elif a == 1: # 1 state    #X
            X | b
        elif a == 2: # + state    #H
            H | b
        elif a == 3: # - state    #XH
            X | b
            H | b
        else :
            print("Create random bits ERROR!!")
    return res_state, qlist



'''
Measure qubits according to specific basis.
input:
    A list of 0/1 indicating bisis.
    A qubit list to be measure.
output:
    A list of results of measurement.
'''
def Measure_by_basis(basisList,qList):
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


'''
To decide whether the challenger passes the challenge.

input:
    Numbers of qubits in list.
    The challenge given to challenger.
    The state list this party has recorded.
    Measurement results from another party.
output:
    The rate of successfully pass the challenge.
'''
def Match_rate_calculate(num_bits,challenge,stateList,res_measure,threshold=0.99):
    False_count=0
    for a,b,c in zip(challenge,stateList,res_measure):
        if int(a)==0 :
            if b<=1 and c[1]>=threshold and b!=c[0]:
                False_count+=1
            elif b>1 and c[1]>=threshold:
                False_count+=1
            else:
                pass
        elif int(a)==1:
            if b>1 and c[1]>=threshold and b-2!=c[0]:
                False_count+=1
            elif b<=1 and c[1]>=threshold:
                False_count+=1
            else:
                pass
        else:
            print("ERROR in challenge value!!")
            return 0
    if num_bits!=0:
        return 1.0-False_count/num_bits
    else:
        return 0
    
    


# In[14]:


class QuantumToken(Protocol):
    
    
    # QuantumToken functions ================================================
    def B_prepare_send(self, num_bits):
        self.stateList, tokenQlist=Create_random_qubits(num_bits)
        self.node_B.ports["portQ_B"].tx_output(tokenQlist)


        
    def A_ask_challenge(self, qList):
        self.tokenQlist=qList.items
        self.A_memory.put(self.tokenQlist)
        
        # schedule waiting for event
        
        My_waitENVtype = EventType("WAIT_EVENT", "Wait for N nanoseconds")
        self._schedule_after(self.waitTime, My_waitENVtype) 
        self._wait_once(ns.EventHandler(self.CpopMem),entity=self
            ,event_type=My_waitENVtype) 
        
    
    def CpopMem(self,event):
        # pop out from qmem
        self.tokenQlist=self.A_memory.pop(list(np.arange(len(self.tokenQlist)))) #pop all
        message = "10101"    #use 10101 as request of challenge
        self.node_A.ports["portC_A1"].tx_output(message)

        
    def B_send_challenge(self,message):
        #print("B_send_challenge at ",ns.sim_time())
        message=message.items[0]
        if str(message)=="10101": #use 10101 as request of challenge
            self.challenge=Random_basis_gen(self.num_bits)
            self.node_B.ports["portC_B2"].tx_output(self.challenge)
        else:
            pass
        
        
    def A_measure_send(self, basisList):
        res_measure = Measure_by_basis(basisList.items,self.tokenQlist)
        self.node_A.ports["portC_A3"].tx_output(res_measure)
        
        
        
    def B_evaluate_reply(self,res_measure):      
        self.success_rate = Match_rate_calculate(self.num_bits,self.challenge
            ,self.stateList,res_measure.items,self.threshold)
        #print("success_rate: ",self.success_rate)
        if self.validation_threshold <= self.success_rate:
            #print("Accepted!")
            self.permission = True
        else:
            #print("Aborted!")
            self.permission = False
    
    
    # basic functions ========================================================================
    def __init__(self, num_bits=8,fiberLenth=1,depolar_rate=0
                 ,timeIND=False,threshold=0.95,waitTime=0): 
        super().__init__()
        self.node_A = Node("A",ID=0,port_names=["portQ_A","portC_A1"
            ,"portC_A2","portC_A3"])
        self.node_B = Node("B",ID=1,port_names=["portQ_B","portC_B1"
            ,"portC_B2","portC_B3"])
        self.MyQfiber = None
        self.fiberLenth = fiberLenth
        self.num_bits = num_bits
        self.stateList = None
        self.tokenQlist = None
        self.challenge = [] 
        self.validation_threshold = threshold
        self.permission = False
        self.A_memory = None
        self.success_rate = None
        self.depolar_rate = depolar_rate
        self.timeIND = timeIND
        self.waitTime = waitTime
        self.threshold=threshold
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
            loss_model=None) 
         
        
        # one directional
        self.node_B.connect_to(self.node_A, self.MyQfiber,
            local_port_name="portQ_B", remote_port_name="portQ_A")

        
        # create classical fibre
        for i in range(1,4):
            MyCfiber=DirectConnection("CFibreConn_"+str(i),
                ClassicalFibre("CFibre_A->B"+str(i), length=self.fiberLenth),
                ClassicalFibre("CFibre_B->A"+str(i), length=self.fiberLenth))
            #create unidirectional fiber dedicated to certain phase of protocol
            if i%2 == 1 : 
                self.node_A.connect_to(self.node_B, MyCfiber,
                    local_port_name="portC_A"+str(i)
                    ,remote_port_name="portC_B"+str(i)) # A to B
            else:
                self.node_B.connect_to(self.node_A, MyCfiber,
                    local_port_name="portC_B"+str(i)
                    ,remote_port_name="portC_A"+str(i)) # B to A
        

        # quantum memory========================================================
        self.A_memory = QuantumMemory("QMemory_A", self.num_bits,ID=0, 
            memory_noise_models=DepolarNoiseModel(depolar_rate=self.depolar_rate,
            time_independent=self.timeIND))
        self.node_A.add_subcomponent(self.A_memory)
        
        B_memory =QuantumMemory("QMemory_B", self.num_bits,ID=1, 
            memory_noise_models=DepolarNoiseModel(depolar_rate=0.5))
        self.node_B.add_subcomponent(B_memory)
        

        #set callback functions===================================================
        self.node_A.ports["portQ_A"].bind_input_handler(self.A_ask_challenge)
        self.node_B.ports["portC_B1"].bind_input_handler(self.B_send_challenge)
        self.node_A.ports["portC_A2"].bind_input_handler(self.A_measure_send)
        self.node_B.ports["portC_B3"].bind_input_handler(self.B_evaluate_reply)
        
        
        # start with B
        self.B_prepare_send(self.num_bits)
        
            



# In[15]:


def run_QuantumToken_sim(run_times=1,fiberLenth=10**-6
    ,num_bits=8,depolar_rate=0,timeIND=False,threshold=0.95,waitTime=0): 
    
    
    MyQuantumTokenList=[]
    for i in range(run_times): 
        ns.sim_reset()
        MyQT=QuantumToken(num_bits=num_bits,fiberLenth=fiberLenth,
            depolar_rate=depolar_rate,timeIND=timeIND,threshold=threshold
            ,waitTime=waitTime)

        ns.sim_run()
        # the success_rate is calculated only before run
        MyQuantumTokenList.append(MyQT.success_rate) 
        
    #ns.logger.setLevel(1)
    return sum(MyQuantumTokenList)/len(MyQuantumTokenList)




# In[16]:


import matplotlib.pyplot as plt

#threshold doesn't matter in this plot
def QuantumToken_plot():
    y_axis=[]
    x_axis=[]
    run_times=10
    num_bits=40
    min_dis=0
    max_dis=10**8

    # first curve
    for i in range(min_dis,max_dis,5*10**6): # from 0 to 10**8 ns
        x_axis.append(i*10**-9) # relate to unit
        y_axis.append(run_QuantumToken_sim(run_times,num_bits
            ,depolar_rate=5,timeIND=False,threshold=0.95,waitTime=i)) 
        
    plt.plot(x_axis, y_axis, 'r-',label='depolar_rate=5')
    
    
    y_axis.clear() 
    x_axis.clear()
    
    # second curve
    for i in range(min_dis,max_dis,5*10**6):
        x_axis.append(i*10**-9) # relate to unit  
        y_axis.append(run_QuantumToken_sim(run_times,num_bits
            ,depolar_rate=1000,timeIND=False,threshold=0.95,waitTime=i)) 
        
    plt.plot(x_axis, y_axis, 'b-',label='depolar_rate=1000')
    
    
    
    plt.title('Quantum Token')
    plt.ylabel('average successful rate for each try')
    plt.xlabel('Alice waiting time (s)') #µ

    plt.legend()
    plt.savefig('QTplotN1.png')
    plt.show()

    

QuantumToken_plot()




