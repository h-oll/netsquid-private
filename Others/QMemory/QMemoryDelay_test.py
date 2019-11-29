#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.protocols import Protocol
from netsquid.components import QuantumMemory, DepolarNoiseModel #,DephaseNoiseModel
from netsquid.qubits import create_qubits
from netsquid.pydynaa import Entity,EventHandler,EventType
from netsquid.components.component import Component, Message
from netsquid.qubits.qformalism import *


# In[7]:


class QMemoryDelay(Protocol):
    
    
    # base functions =======================================================
    def __init__(self,num_bits=8,depolar_rate=0,timeIND=False,delay=0): 
        super().__init__()
        set_qstate_formalism(QFormalism.DM)
        self.my_memory = QuantumMemory("QMemory", num_bits,ID=0, 
            memory_noise_models=DepolarNoiseModel(depolar_rate=depolar_rate,
            time_independent=timeIND))     
        self.num_bits=num_bits
        self.delay=delay
        self.start()
        self.qList=[]
        
        
        
    def stop(self):
        super().stop()
        self._running = False
        
    def is_connected(self):
        super().is_connected()
        pass
        
    
    def start(self):
        super().start()
        My_waitENVtype = EventType("WAIT_EVENT", "Wait for 8 nanoseconds")
        self._schedule_after(self.delay, My_waitENVtype) # self.delay
        self._wait_once(ns.EventHandler(self.popMem),entity=self
            ,event_type=My_waitENVtype) # can't add event_type
        
        
        
    # my functions ============================================
    def popMem(self,event):
        print("time: ",ns.sim_time())
        # pop out from qmem
        self.qList=self.my_memory.pop(list(np.arange(self.num_bits)))
         


# In[8]:


ns.sim_reset()

num_bits=1
depolar_rate=1000
qList = create_qubits(num_bits,system_name="Q")


qt=QMemoryDelay(num_bits=num_bits,depolar_rate=depolar_rate,delay=99999) #test
qt.my_memory.put(qList)   # put in qmem


#ns.logger.setLevel(1)
ns.sim_run()

for q in qt.qList:
    print(q.qstate.dm)


# In[ ]:




