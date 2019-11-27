#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:



def Random_basis_gen(num_bits):
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
    return opList

    
# returns a list of stats and a list of qubits
def Create_random_qubits(num_bits):
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


def Match_rate_calculate(num_bits,challenge,stateList,res_measure):
    False_count=0
    for a,b,c in zip(challenge,stateList,res_measure):
        if int(a)==0 :
            if b<=1 and c[1]>=0.999 and b!=c[0]:
                False_count+=1
                #print(a,b,c)
            elif b>1 and c[1]>=0.999:
                False_count+=1
                #print(a,b,c)
            else:
                pass
        elif int(a)==1:
            if b>1 and c[1]>=0.999 and b-2!=c[0]:
                False_count+=1
                #print(a,b,c)
            elif b<=1 and c[1]>=0.999:
                False_count+=1
                #print(a,b,c)
            else:
                pass
        else:
            print("ERROR in challenge value!!")
            return 0
        
    if num_bits!=0:
        return 1.0-False_count/num_bits
    else:
        return 0
    
    


# In[4]:


class QuantumToken(Protocol):
    
    
    # QuantumToken functions ================================================
    def B_prepare_send(self, num_bits):
        self.stateList, tokenQlist=Create_random_qubits(num_bits)
        self.node_B.ports["portQ_B"].tx_output(tokenQlist)


        
    def A_ask_challenge(self, qList):
        self.tokenQlist=qList.items
        self.A_memory.put(self.tokenQlist)
        #print("sleeping...")
        #time.sleep(5)   
        temp=[]
        for i in range(len(self.tokenQlist)):
            temp.append(i)
        self.tokenQlist=self.A_memory.pop(temp)
        message="10101"    #use 10101 as request of challenge
        self.node_A.ports["portC_A1"].tx_output(message)
        
        
    def B_send_challenge(self,message):
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
        self.success_rate=Match_rate_calculate(self.num_bits,self.challenge
            ,self.stateList,res_measure.items)
        #print("success_rate: ",self.success_rate)
        
        if self.validation_threshold<=self.success_rate:
            #print("Accepted!")
            self.permission = True
        else:
            #print("Aborted!")
            self.permission = False
    
    
    # basic functions ========================================================================
    def __init__(self, num_bits=8,fiberLenth=1,depolar_rate=0,timeIND=False): 
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
        self.challenge = []               # B to A
        self.validation_threshold = 0.95
        self.permission = False
        self.A_memory = None
        self.success_rate = None
        self.depolar_rate = depolar_rate
        self.timeIND = timeIND
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
            loss_model=None, 
            noise_model="default") 
         
        
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
            time_independent=self.timeIND))#True
        #QuantumNoiseModel
        #DephaseNoiseModel(dephase_rate
        #DepolarNoiseModel(depolar_rate
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
        
            



# In[5]:


def run_QuantumToken_sim(run_times=1,fiberLenth=10**-6
    ,num_bits=8,depolar_rate=0,timeIND=False):
    
    MyQuantumTokenList=[]
    for i in range(run_times): 
        #print("The ",i+1,"th run...")
        ns.sim_reset()
        MyQT=QuantumToken(num_bits=num_bits,fiberLenth=fiberLenth,
            depolar_rate=depolar_rate,timeIND=timeIND)

        ns.sim_run()
        #the success_rate is calculated only before run
        MyQuantumTokenList.append(MyQT.success_rate) 
        
    #ns.logger.setLevel(1)
    return MyQuantumTokenList




# In[7]:


import matplotlib.pyplot as plt

def QuantumToken_plot():
    y_axis=[]
    x_axis=[]
    run_times=50
    num_bits=40
    min_dis=0
    max_dis=100

    #first line
    for i in range(min_dis,max_dis,1):
        #rate_sum=0.0
        x_axis.append(i/100)
        
        rate_list=run_QuantumToken_sim(run_times,num_bits
            ,depolar_rate=i/100,timeIND=True)
        
        y_axis.append(sum(rate_list)/len(rate_list))
        
    plt.plot(x_axis, y_axis, 'go-',label='MemNoiseModel1')
    
    #y_axis.clear() 
    #x_axis.clear()

        
    plt.ylabel('average successful rate')
    plt.xlabel('depolar rate')

    plt.legend()
    plt.savefig('QTplot3.png')
    plt.show()

    

QuantumToken_plot()


# In[ ]:




