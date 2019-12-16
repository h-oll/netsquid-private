#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import netsquid as ns
from netsquid.qubits.qubitapi import *

'''
function:
    Generate a GHZ set with customized length and entangled qubits.  

input:
    num_qubits: Numbers of qubits in a column.
    num_sets: Numbers of qubits in a raw.

output:
    A 2-D arrary of qubits with every one in the same raw entangled.
'''

def Create_GHZ_set_list(num_qubits,num_sets):
    qList_2D=[]
    
    for i in range(num_qubits):
        qList = create_qubits(num_sets) #qubit 000
        H | qList[0]
        tmp=[qList[0]]
        for j in range(1, num_sets):
            ns.qubits.operate([qList[0],qList[j]], ns.CNOT)
            tmp.append(qList[j])

        qList_2D.append(tmp)

    return qList_2D


# In[1]:


#Verify

from netsquid.qubits.operators import *
tmp=Create_GHZ_set_list(5,3)
print(tmp)

np.asarray(tmp)
print(tmp)
print(tmp[:,2])



mes0=ns.qubits.qubitapi.measure(tmp[2][0],observable=Z)
mes1=ns.qubits.qubitapi.measure(tmp[2][1],observable=Z) 
mes2=ns.qubits.qubitapi.measure(tmp[2][2],observable=Z) 

print(mes0)
print(mes1)
print(mes2)


# In[108]:


'''
function:
    Generate a random serial number list.  

input:
    num_qubits: Length of serial number.
    min: Minimum value possible in the list.
    max: Maximum value possible in the list.

output:
    A random serial number list.
'''
from random import randint

def SerialNumGen(num_bits,min,max):
    #seed() 
    bitList=[]
    startUSN=randint(min,max-num_bits+1)
    for i in range(num_bits):
        bitList.append(startUSN)
        startUSN+=1
    return bitList


# In[114]:


#verify
SerialNumGen(7,0,10)


# In[12]:


'''
function:
   One way function which can be used in many place.  

input:
    any

output:
    A qubit in this case.
'''

from netsquid.qubits import create_qubits
from netsquid.qubits.operators import *

# hash with the symmetric key, Unique Serial Number, the amound of money
def OneWayFunction(identity=None,symkey=[],randomSerialNumber=0,Money=0):
    owf_key=''
    
    # covert inputs to binary
    for i in symkey:
        owf_key+=str(bin(i)[2:])
    owf_key+=str(bin(randomSerialNumber)[2:])
    owf_key+=str(bin(Money)[2:]) 
    owf_key=int(owf_key)
    
    # make it qubit
    # apply three big prime numbers
    p1 = 33179
    p2 = 32537
    p3 = 31259
    
    MyRx=create_rotation_op(np.pi/180*(owf_key%p1), (1, 0, 0))
    MyRy=create_rotation_op(np.pi/180*(owf_key%p2), (0, 1, 0))
    MyRz=create_rotation_op(np.pi/180*(owf_key%p3), (0, 0, 1))
    
    tempQubit=create_qubits(1)
    tempQubit=tempQubit[0]
    MyRx | tempQubit
    MyRy | tempQubit
    MyRz | tempQubit
    
    #print(tempQubit.qstate.dm)
    return tempQubit


# In[ ]:


'''
function:
    Cswap function.
input:
    Three qubits.
output:
    Three qubits applied Cswap.
'''
# C swap can be composed by T,H
# see https://www.mathstat.dal.ca/~selinger/quipper/doc/QuipperLib-GateDecompositions.html
from netsquid.qubits.operators import H,T
def Cswap(qA,qB,qC):
    
    invT=T.inv
    
    operate([qC, qB], ops.CNOT)
    H | qC
    T | qA
    T | qB
    T | qC
    operate([qB, qA], ops.CNOT)
    operate([qC, qB], ops.CNOT)
    operate([qA, qC], ops.CNOT)
    T | qC
    invT | qB
    operate([qA, qB], ops.CNOT)
    invT | qA
    invT | qB
    operate([qC, qB], ops.CNOT)
    operate([qA, qC], ops.CNOT)
    operate([qB, qA], ops.CNOT)
    H | qC
    operate([qC, qB], ops.CNOT)
    return qA,qB,qC


# In[11]:


'''
function:
    Swap test which exames the closeness of two qubits.

input:
    two qubits.

output:
    A tuple indecating the index and pobability.
    (0,0.5) means orthogonal.
    (0,1)   means the two are equal.
'''
from netsquid.qubits import create_qubits
from netsquid.qubits.operators import H,Z

def SwapTest(qB,qC):
    qA=create_qubits(1)
    qA=qA[0]
    H | qA
    Cswap(qA,qB,qC)
    H | qA
    return ns.qubits.qubitapi.measure(qA,observable=Z)


# In[9]:


'''
function:
    Create qubits list.

input:
    numbers of qubits.

output:
    A list of quantum states.(0,1,+,-)
    And corespond quantum list.
'''

from netsquid.qubits import create_qubits
from random import randint
from netsquid.qubits.operators import H,X

def Create_random_qubits(num_bits):
    res_state=[]
    qlist=[]
    qlist=create_qubits(num_bits) 
    for i in range(0,num_bits):
        res_state.append(randint(0,3)) # in four states
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


# In[13]:


'''
function:
    Measuring qubits according to certain basis.
    Names of qubits need to be indexed from 0

input:
    A list of basis consised by 0/1. (0:standard, 1:Hadamard)
    A list of qubits.

output:
    A list of measurment tuple accordingly. Return merely 0 means missing such qubits
'''

import netsquid as ns

def Measure_by_basis(basisList,qList):
    if len(basisList)<len(qList): 
        print("Quantum list is too long! ERROR!!")
        return 0
    else:
        res_measurement=[0]*len(basisList) #init to 0
        
        for q in qList:
            pos=int(q.name[5:]) #get qubit index #defalt first qubit name = QS#0-0
            if basisList[pos]==0:
                res_measurement[pos]=ns.qubits.qubitapi.measure(q,observable=Z) #measure in standard basis
            elif basisList[a]==1:
                res_measurement[pos]=ns.qubits.qubitapi.measure(q,observable=X) #measure in Hadamard basis
            else:
                print("measuring ERROR!!\n")    
        return res_measurement


# In[ ]:


'''
function:
    Wait certain amout of simulated time in simulation 
    This is the way NetSquid implements waiting action in simulated time.
    By customizing a wait event, it will call End_waiting function after waiting.
    More example at https://github.com/h-oll/netsquid-private/blob/master/Others/QMemory/QMemoryNoiceSim.py
'''
class example_class():
    
    def example_function:
        # Put folowing lines in functions you want to wait.
        My_waitENVtype = EventType("WAIT_EVENT", "Wait for N nanoseconds")
        self._schedule_after(customized_delay, My_waitENVtype) # customized_delay
        self._wait_once(ns.EventHandler(self.End_waiting),entity=self,event_type=My_waitENVtype) 
        # Put above lines in functions you want to wait.
        
    # called after qaiting
    def End_waiting(self,event):
        #continue your protocol
         

