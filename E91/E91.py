#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.components import ClassicalFibre,QuantumFibre
from netsquid.qubits import create_qubits
from netsquid.qubits.operators import H,Z,X
from netsquid.protocols import Protocol
from netsquid.nodes.connections import  DirectConnection

from random import randint


# In[50]:


'''
Create EPR pairs.
input:
    Numbers of pairs.
output:
    Two lists of qubits, with the corresponding slots entangled.
'''
def Create_multiEPR(num_bits):
    qListA=[]
    qListB=[]
    for i in range(num_bits):
        qA, qB = create_qubits(2) # qubit 00
        ns.qubits.operate(qA, ns.H)
        ns.qubits.operate([qA,qB], ns.CNOT)
        qListA.append(qA)
        qListB.append(qB)
    return qListA, qListB 



'''
Randomly measure a qubits list by Z or X basis.
Input:
    Numbers of qubits that should be >= the length of qlist. Equal case happens when no loss.
    Qubit list to measure.
Output:
    basisList: A list of basis applied(Z X -1). -1 means qubit missing. (detect by qubit name)
    loc_res_measure: A list of measurment results. If there's a qubit loss, 
    both opList and loc_res_measure will have value -1 in the such slot in the list.

'''

def Random_ZX_measure(num_bits,qlist):
    num_start=int(qlist[0].name[3:-len('-')-1])# get value after qubit name "QS#<i>-n"
    basisList = []*num_bits  # set boundary
    loc_res_measure=[]*num_bits  # set boundary
    ind=0
    for i in range(num_start,num_start+num_bits):
        if ind <= len(qlist)-1:
            if int(qlist[ind].name[3:-len('-')-1]) == i:
                rbit = randint(0,1) # 0:Z 1:X
                if rbit:
                    basisList.append('X')
                    loc_res_measure.append(ns.qubits.qubitapi.
                        measure(qlist[ind],observable=X)[0]) #measure in Hadamard basis
                else:
                    basisList.append('Z')
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
Compare two lists, find the unmatched index, 
    then remove corresponding slots in loc_meas.
Input:
    loc_basis_list,res_basis_list: Two lists with elements 0-2 (Z,X, -1:qubit missing).
    loc_meas: Local measurement result to keep 
Output:
    measurement result left.

'''
def Compare_basis(loc_basis_list,res_basis_list,loc_meas):
    
    if len(loc_basis_list) != len(res_basis_list):
        print("Comparing error! length issue!")
        ''' debug
        print(loc_basis_list)
        print(len(loc_basis_list))
        print(res_basis_list)
        print(len(res_basis_list))
        '''
        return -1
    
    popList=[]
    
    for i in range(len(res_basis_list)):
        if loc_basis_list[i] != res_basis_list[i]:
            popList.append(i)
    
    
    for i in reversed(popList): 
        if loc_meas:
            loc_meas.pop(i)
        
    return loc_meas


# In[51]:


class E91(Protocol):
    
    def E91_A_sendHalf_EPR(self):
        qListA,qlistB = Create_multiEPR(self.num_bits)
        self.qListA = qListA
        self.node_A.ports["portQA"].tx_output(qlistB)
        
        
        
        
    def E91_B_randMeas(self,qListB):
        self.qListB = qListB.items
        #print(qListB)
        self.basisList_B, self.res_measure_B = Random_ZX_measure(self.num_bits,
            self.qListB)
        #print(self.basisList_B)
        self.node_B.ports["portCB_1"].tx_output(self.basisList_B)
        
        
        
        
    def E91_A_compare_keyGen(self,basisList_fromB):
        self.basisList_A, self.res_measure_A = Random_ZX_measure(self.num_bits,
            self.qListA)
        self.node_A.ports["portCA_2"].tx_output(self.basisList_A)
        #print(self.basisList_A)
        #print(basisList_fromB.items)
        self.key_A=Compare_basis(self.basisList_A,basisList_fromB.items,
            self.res_measure_A)
        self.key_A = ''.join(map(str, self.key_A))
        print(self.key_A)  # show results

        
        
        
        
    def E91_B_compare_keyGen(self,basisList_fromA):
        self.key_B = Compare_basis(self.basisList_B,basisList_fromA.items,self.res_measure_B)
        self.key_B = ''.join(map(str, self.key_B))
        print(self.key_B)  # show results
        
        
        
    
    #control functions===========================================================    
    def __init__(self, num_bits=8,fibre_len=10**-6): 
        super().__init__()
        self.node_A = Node("A",ID=0,port_names=["portQA","portCA_1","portCA_2"])
        self.node_B = Node("B",ID=1,port_names=["portQB","portCB_1","portCB_2"])
        self.MyQfiber = None
        self.MyCfiber = None
        self.num_bits = num_bits
        self.qListA = []
        self.qListB = []
        self.basisList_A = []
        self.basisList_B = []
        self.res_measure_A = []
        self.res_measure_B = []
        self.key_A = []
        self.key_B = []
        
        self.fiberLenth = fibre_len
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
        self.MyCfiber=DirectConnection("CFibreConn_B->A",
            ClassicalFibre("CFibre_B->A", length=self.fiberLenth))
        self.MyCfiber2=DirectConnection("CFibreConn_A->B",
            ClassicalFibre("CFibre_A->B", length=self.fiberLenth))
        
        self.node_A.connect_to(self.node_B, self.MyQfiber,
            local_port_name="portQA", remote_port_name="portQB")
        
        self.node_B.connect_to(self.node_A, self.MyCfiber,
            local_port_name="portCB_1", remote_port_name="portCA_1")
        
        self.node_A.connect_to(self.node_B, self.MyCfiber2,
            local_port_name="portCA_2", remote_port_name="portCB_2")
        
        
        #set callback functions===================================================
        self.node_B.ports["portQB"].bind_input_handler(self.E91_B_randMeas)
        self.node_A.ports["portCA_1"].bind_input_handler(self.E91_A_compare_keyGen)
        self.node_B.ports["portCB_2"].bind_input_handler(self.E91_B_compare_keyGen)
        
        
        
        # Alice starts======================================================
        self.E91_A_sendHalf_EPR()        


# In[52]:


# Test
ns.sim_reset()
MyE91=E91(num_bits=70)
ns.sim_run()


# In[ ]:




