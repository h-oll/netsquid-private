#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.components import QuantumMemory, DepolarNoiseModel #,DephaseNoiseModel
from netsquid.qubits import create_qubits


# In[94]:


class QMemoryTest():
    
    
    def init(self,num_bits=8,depolar_rate=0,timeIND=False,waitTime=5):
        self.node_A = Node("A",ID=0)
        #apply quantum memory and Depolar Noise Model
        self.A_memory = QuantumMemory("QMemory_A", num_bits,ID=0, 
            memory_noise_models=DepolarNoiseModel(depolar_rate=depolar_rate,
            time_independent=timeIND))
        self.node_A.add_subcomponent(self.A_memory)
        self.num_bits=num_bits
        self.waitTime=waitTime
        self.qList=None
        return(self.start())
    
    
    def CompareQubits(self,qList):
        tmp=create_qubits(1,system_name="Q") #use one qubit to compare
        err_count=0
        for q in qList:
            if np.array_equal(q.qstate.dm,tmp[0].qstate.dm):
                pass
            else:
                err_count+=1
        return err_count

        
    def start(self):
        self.qList = create_qubits(self.num_bits,system_name="Q") 
        
        self.A_memory.put(self.qList)
        
        # put
        ns.sim_run(duration=self.waitTime,magnitude=ns.MICROSECOND)  
        # pop
        temp=[]
        for i in range(self.num_bits):
            temp.append(i)
        self.qListB=self.A_memory.pop(temp)
        
        err_count=self.CompareQubits(self.qList)
        
        if self.num_bits!=0:
            return err_count/self.num_bits
        else:
            return 0


# In[95]:


def run_QMemoryTest(run_times=1,num_bits=8,depolar_rate=0,
        timeIND=False,waitTime=0):
    MyQTList=[]
    sum=0
    for i in range(run_times):
        ns.sim_reset()
        MyQTList.append(QMemoryTest())
        n=MyQTList[i].init(num_bits=num_bits,depolar_rate=depolar_rate
            ,timeIND=timeIND,waitTime=waitTime)#
        ns.sim_run()
        sum+=n
    return sum/run_times
    
    


# In[102]:


import matplotlib.pyplot as plt

def QuantumMem_plot():
    y_axis=[]
    x_axis=[]
    run_times=30
    num_bits=40
    depolar_rate=500
    timeIND=False
    waitTime=0
    
    min_time=0
    max_time=7000
    
    #first line
    for i in range(min_time,max_time,50):
        
        x_axis.append(i)
        y_axis.append(run_QMemoryTest(run_times=run_times
            ,num_bits=num_bits,depolar_rate=depolar_rate
            ,timeIND=timeIND,waitTime=i))
        
    plt.plot(x_axis, y_axis, 'go-',label='depolar rate = 500')
    
    #second line
    x_axis.clear()
    y_axis.clear()
    depolar_rate=1000
    
    for i in range(min_time,max_time,50):
        
        x_axis.append(i)
        y_axis.append(run_QMemoryTest(run_times=run_times
            ,num_bits=num_bits,depolar_rate=depolar_rate
            ,timeIND=timeIND,waitTime=i))
        
    plt.plot(x_axis, y_axis, 'bo-',label='depolar rate = 1000')
    
    
    plt.title('Depolar Noise Effects on qubits')
    plt.ylabel('average qubit error rate ')
    plt.xlabel('time stayed in quantum memory (μs)')

    plt.legend()
    plt.savefig('QMem4.png')
    plt.show()



QuantumMem_plot()

