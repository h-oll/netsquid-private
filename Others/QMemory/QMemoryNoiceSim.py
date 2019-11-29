


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.components import QuantumMemory, DepolarNoiseModel
from netsquid.qubits import create_qubits
from netsquid.qubits.qformalism import *





class QMemoryTest():
    
    
    def init(self,num_bits=8,depolar_rate=0,timeIND=False,waitTime=5):
        set_qstate_formalism(QFormalism.DM)
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
    

    
    def CompareQubits(self,qubit):
        #print(float(qubit.qstate.dm[1][1]))
        return float(qubit.qstate.dm[1][1])

        
    def start(self):
        
        self.qList = create_qubits(self.num_bits,system_name="Q") 
        
        self.A_memory.put(self.qList)
        
        # put
        ns.sim_run(duration=self.waitTime,magnitude=ns.MICROSECOND)  
        
        # pop            
        self.qList=self.A_memory.pop(list(np.arange(self.num_bits)))
        
        err_rate=self.CompareQubits(self.qList[0])
        
        if self.num_bits!=0:
            return err_rate   
        else:
            return 0




import matplotlib.pyplot as plt

def QuantumMem_plot():
    x_axis=[]
    y_axis=[]
    run_times=1
    num_bits=1
    timeIND=False
    waitTime=0
    depolar_rate=1000
    
    
    min_time=0
    max_time=7000
    
    ns.sim_reset()
    
    
    #first line
    for i in range(min_time,max_time,50):
        
        x_axis.append(i)
        y_axis.append(QMemoryTest().init(num_bits=num_bits
            ,depolar_rate=depolar_rate,timeIND=timeIND,waitTime=i))
     
        
    plt.plot(x_axis, y_axis, 'go-',label='depolar rate = 1000')
    
    
    
    #second line
    x_axis.clear()
    y_axis.clear()
    depolar_rate=500
    
    for i in range(min_time,max_time,50):
        
        x_axis.append(i)
        y_axis.append(QMemoryTest().init(num_bits=num_bits
            ,depolar_rate=depolar_rate,timeIND=timeIND,waitTime=i))
        
    plt.plot(x_axis, y_axis, 'bo-',label='depolar rate = 500')
    
    
    
    # 3rd line
    x_axis.clear()
    y_axis.clear()
    depolar_rate=50
    
    for i in range(min_time,max_time,50):
        
        x_axis.append(i)
        y_axis.append(QMemoryTest().init(num_bits=num_bits
            ,depolar_rate=depolar_rate,timeIND=timeIND,waitTime=i))
        
    ns.sim_run()
    
    
    plt.plot(x_axis, y_axis, 'ro-',label='depolar rate = 50')
    
    
    
    plt.title('Depolar Noise Effects on qubits')
    plt.ylabel('average qubit error rate ')
    plt.xlabel('time stayed in quantum memory (μs)')

    plt.legend()
    plt.savefig('QMem7.png')
    plt.show()



QuantumMem_plot()






