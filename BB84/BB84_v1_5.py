#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.nodes.connections import DirectConnection
from netsquid.components import ClassicalFibre,QuantumFibre,FibreLossModel
from netsquid.qubits import create_qubits
from netsquid.qubits.operators import *
from random import seed , randint
from netsquid.protocols import Protocol
from netsquid.components.models import FixedDelayModel, GaussianDelayModel, FibreDelayModel


# In[2]:


#====================================other functions===========================================
# if there's a qubit loss, both opList and loc_res_measure will have 2 in the slot
def random_ZX_measure(num_bits,qlist):
    seed(randint(0, 2**num_bits))
    opList = [2]*num_bits
    
    loc_res_measure=[2]*num_bits
    
    for q in qlist:
        rbit = randint(0,1)
        a=int(q.name[1:])
        opList[a] = rbit
        if rbit==0:
            loc_res_measure[a]=ns.qubits.qubitapi.measure(q,observable=Z) #measure in standard basis
        elif rbit==1:
            loc_res_measure[a]=ns.qubits.qubitapi.measure(q,observable=X) #measure in Hadamard basis
        else:
            print("measuring ERROR!!\n")
    return opList,loc_res_measure

    
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


def Compare_measurement(num_bits,stateList,opList):
    matchList=[]
    for i in range(0,num_bits):
        if stateList[i]<2 and opList[i]==0:    #measure in standard basis
            matchList.append(i)
        elif stateList[i]>=2 and opList[i]==1: #measure in Hadamard basis
            matchList.append(i)
        else:
            pass
    return matchList



# In[3]:


class BB84(Protocol):
    
    #BB84 functions===========================================================   
    
    def BB84_Alice_sendQubits(self):
        self.stateList,qlist=create_random_qubits(self.num_bits)
        self.node_A.ports["portQA"].tx_output(qlist)
        
    
    def BB84_Bob_measure_send(self,qlist): # some qubits might be lost
        qlist=qlist.items
        
        if isinstance(qlist[0], int):
            pass
        else:
            #qlist=Add_dummy(self.num_bits,qlist) #to compensate qubit loss
            # B measuring
            B_basis, self.res_measure=random_ZX_measure(self.num_bits,qlist) 
            if B_basis == -1:
                print("B measuring failed!!")
            else :
                # B send measurement
                self.portCB.tx_output(B_basis)  

    
    
    def BB84_Alice_measure_send_keygen(self,opList):
        opList=opList.items
        
        # A measure 
        matchList=Compare_measurement(self.num_bits,self.stateList,opList) #get opList from B

        # A return  matchList to B
        self.portCA.tx_output(matchList)
        
        for i in matchList:
            self.key_A.append(self.stateList[i]%2) #quantum state 0,+:0    1,-:1
        #print("key_A=",self.key_A)
        return self.key_A
    
    
    
    def BB84_Bob_keygen(self,matchList):
        matchList=matchList.items
        
        for i in matchList:
            self.key_B.append(self.res_measure[int(i)][0])
        #print("key_B=",self.key_B)
        return self.key_B

    
    #control functions===========================================================    
    def __init__(self, num_bits=8,fibre_len=10**-6,fibre_loss_init=0.2,fibre_loss_length=0.25): #,fibreLossModel=default
        super().__init__()
        self.node_A = Node("A",ID=0,port_names=["portQA","portCA"])
        self.node_B = Node("B",ID=1,port_names=["portQB","portCB"])
        self.portCB = self.node_B.ports["portCB"] #classical ports are not inherited by nodes
        self.portCA = self.node_A.ports["portCA"]
        self.MyQfiber = None
        self.MyCfiber = None
        self.num_bits = num_bits
        self.stateList = None
        self.res_measure = None
        self.key_A = []
        self.key_B = []
        self.fiberLenth = fibre_len
        self.fibre_loss_init = fibre_loss_init
        self.fibre_loss_length = fibre_loss_length
        self.start()
    
    
    def stop(self):
        super().stop()
        self._running = False
        
    def is_connected():
        super().is_connected()
        pass
        
        
    def start(self):
        super().start()

        
        # connect and connect quantum fibres
        self.MyQfiber=QuantumFibre("QFibre_A->B", length=self.fiberLenth, 
            loss_model=FibreLossModel(p_loss_length=self.fibre_loss_length,p_loss_init=self.fibre_loss_init), 
            depolar_rate=0, noise_model="default") 
        #FibreLossModel(p_loss_length=0.2,p_loss_init=0.83)
        

        # create classical fibre
        self.MyCfiber=DirectConnection("CFibreConn_A->B",
            ClassicalFibre("CFibre_A->B", length=self.fiberLenth),
            ClassicalFibre("CFibre_B->A", length=self.fiberLenth))
        
        self.node_A.connect_to(self.node_B, self.MyQfiber,
            local_port_name="portQA", remote_port_name="portQB")
        
        self.node_B.connect_to(self.node_A, self.MyCfiber,
            local_port_name="portCB", remote_port_name="portCA")
        

        #set callback functions===================================================
        self.node_B.ports["portQB"].bind_input_handler(self.BB84_Bob_measure_send)
        self.node_A.ports["portCA"].bind_input_handler(self.BB84_Alice_measure_send_keygen)
        self.node_B.ports["portCB"].bind_input_handler(self.BB84_Bob_keygen)
        
        
        # Alice starts======================================================
        self.BB84_Alice_sendQubits()
        


# In[4]:


#===========================================execution=========================================================
def run_BB84_sim(runtimes=1,num_bits=8,fibre_len=10**-6,fibre_loss_init=0.2,fibre_loss_length=0.25):
    ns.sim_reset()
    
    run_times=runtimes
    MyBB84List=[]  #protocol list
    
    for i in range(run_times): 
        
        #print("The ",i,"th run...")
        MyBB84List.append(BB84(num_bits,fibre_len,fibre_loss_init,fibre_loss_length).key_B)
        
    ns.sim_run()
    
    #print("MyBB84List:",MyBB84List)
    
    return MyBB84List

#ns.set_console_debug()
#run_BB84_sim(5,8,10**-18)


# In[6]:


#===========================================plot======================================================
import matplotlib.pyplot as plt

def BB84_plot():
    y_axis=[]
    x_axis=[]
    run_times=1
    num_bits=40
    min_dis=100
    max_dis=100000

    #first line
    for i in range(min_dis,max_dis,1000):
        key_sum=0.0
        x_axis.append(1.*i/1000) 
        key_list=run_BB84_sim(run_times,num_bits,1.*i/1000)
        for j in key_list:
            key_sum=key_sum+len(j)
        y_axis.append(key_sum/run_times/num_bits)
        
    plt.plot(x_axis, y_axis, 'go-',label='FibreLossModel1')
    
    y_axis.clear() 
    x_axis.clear()
    
    #second line
    for i in range(min_dis,max_dis,1000):
        key_sum=0.0
        x_axis.append(1.*i/1000) 
        #FibreLossModel(p_loss_init=0.83,p_loss_length=0.2)
        key_list=run_BB84_sim(run_times,num_bits,1.*i/1000,fibre_loss_init=0.83,fibre_loss_length=0.2)
        for j in key_list:
            key_sum=key_sum+len(j)
        y_axis.append(key_sum/run_times/num_bits)

    plt.plot(x_axis, y_axis, 'bo-',label='FibreLossModel2')
        
        
    plt.ylabel('average key length/original qubits length')
    plt.xlabel('fibre lenth (km)')

    
    #plt.xscale('log')
    plt.legend()
    plt.savefig('plot.png')
    plt.show()

    

#BB84_plot()

