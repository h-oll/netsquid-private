#!/usr/bin/env python
# coding: utf-8

# In[13]:


import netsquid as ns
from netsquid.components.qmemory import QuantumMemory
from netsquid.components import ClassicalFibre,QuantumFibre
from netsquid.nodes.connections import  DirectConnection
from netsquid.nodes import Node
from netsquid.qubits import operators as ops
from netsquid.qubits import create_qubits
from netsquid.qubits.operators import Z,H,X,Y
from netsquid.protocols import Protocol
from netsquid.qubits.qubitapi import measure


# In[14]:


def Creat_EPR():
    # creat qubits
    q0, q1 = create_qubits(2) #qubit 00
    # entangle the two
    # do Hadmard transform
    ns.qubits.operate(q0, ns.H)
    # do cnot operation
    ns.qubits.operate([q0,q1], ns.CNOT) 
   
    return q0,q1 


# In[15]:


class QuantumTeleportation(Protocol):
    
    def QT_Alice_EPRGen_sendQubits(self):
        self.qubit_A,q1 =Creat_EPR()
        self.node_A.ports["portQA"].tx_output(q1)
        
        
    def QT_Bob_rec_Alice_measure(self,qubit):
        self.qubit_B=qubit.items[0]
        # A measuere by bell state
        ns.qubits.operate([self.qubit_A, self.qubitToCopy], ops.CNOT)     
        H | self.qubit_A
        res_bellMeas=[]
        res_bellMeas.append(measure(self.qubitToCopy,observable=Z)[0])
        res_bellMeas.append(measure(self.qubit_A,observable=Z)[0])
        
        self.node_A.ports["portCA"].tx_output(res_bellMeas)
        
        
    def QT_Bob_dup(self,bellRes):
        bellRes=bellRes.items
        if bellRes[0] == 0   and bellRes[1] == 1:
            Z | self.qubit_B
        if bellRes[0] == 1   and bellRes[1] == 0:
            X | self.qubit_B
        if bellRes[0] == 1   and bellRes[1] == 1:
            Y | self.qubit_B #Y=XZ
        
        # show result
        print("copied:\n",self.qubit_B.qstate.dm)
        
        
    def __init__(self,qubitToCopy,fiberLenth=10**-6): 
        #set_qstate_formalism(QFormalism.DM)
        self.fiberLenth=fiberLenth
        self.qubitToCopy=qubitToCopy
        self.qubit_A=None
        self.qubit_B=None
        self.node_A = Node("A",ID=0,port_names=["portQA","portCA"])
        self.node_B = Node("B",ID=1,port_names=["portQB","portCB"])
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
        self.MyQfiber=QuantumFibre("QFibre_A->B", length=self.fiberLenth)
        
        # create classical fibre
        self.MyCfiber=DirectConnection("CFibreConn_A->B",
            ClassicalFibre("CFibre_A->B", length=self.fiberLenth))
        
        self.node_A.connect_to(self.node_B, self.MyQfiber,
            local_port_name="portQA", remote_port_name="portQB")
        
        self.node_A.connect_to(self.node_B, self.MyCfiber,
            local_port_name="portCA", remote_port_name="portCB")
        


        #set callback functions===================================================
        self.node_B.ports["portQB"].bind_input_handler(self.QT_Bob_rec_Alice_measure)
        self.node_B.ports["portCB"].bind_input_handler(self.QT_Bob_dup)
        
        
        # Alice starts======================================================
        self.QT_Alice_EPRGen_sendQubits()






# In[16]:


# Test
ns.sim_reset()
q=create_qubits(1)
q=q[0]
H|q
Z|q

print("origin: \n",q.qstate.dm)
myQT=QuantumTeleportation(q,10**-6)
ns.sim_run()
    





# In[ ]:




