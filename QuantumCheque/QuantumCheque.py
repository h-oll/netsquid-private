#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from BB84_lib_v1_5 import *
from netsquid.qubits import ketstates
from netsquid.qubits.operators import *
from netsquid.qubits.qubitapi import *
from netsquid.qubits import operators as ops
from netsquid.components import QuantumMemory, DepolarNoiseModel
from netsquid.pydynaa import Entity,EventHandler,EventType

from random import seed, randint


# In[2]:


def Create_GHZ_triple_qubits_list(num_qubits=8):
    qListB=[]
    qListA1=[]
    qListA2=[]
    for i in range(num_qubits):
        qList = create_qubits(3) #qubit 000
        
        ns.qubits.operate(qList[0], ns.H)
        ns.qubits.operate([qList[0],qList[1]], ns.CNOT) 
        ns.qubits.operate([qList[0],qList[2]], ns.CNOT)
        
        qListB.append(qList[0])
        qListA1.append(qList[1])
        qListA2.append(qList[2])

    return qListB,qListA1,qListA2



def UniqueSerialNumGen(num_bits):
    seed() 
    bitList=[]
    startUSN=randint(0,1000)
    for i in range(num_bits):
        bitList.append(startUSN)
        startUSN+=1
    return bitList



# hash with the symmetric key, Unique Serial Number, the amound of money
def OneWayFunction(identity=None,symkey=[],randomSerialNumber=0,Money=0):
    owf_key=''
    for i in symkey:
        owf_key+=str(bin(i)[2:])
        
    owf_key+=str(bin(randomSerialNumber)[2:])
    owf_key+=str(bin(Money)[2:]) 
    owf_key=int(owf_key)
    
    # make it qubit
    # apply three prime numbers
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
    
    
    
    
# C swap can be composed by T
# https://www.mathstat.dal.ca/~selinger/quipper/doc/QuipperLib-GateDecompositions.html
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



def SwapTest(qB,qC):
    qA=create_qubits(1)
    qA=qA[0]
    H | qA
    Cswap(qA,qB,qC)
    H | qA
    return ns.qubits.qubitapi.measure(qA,observable=Z)
    



# In[3]:


class QuantumCheque(Protocol):
  

    # QuantumCheque function =========================================================

    def BA_BB84_keygen(self,num_bits=8,fibre_len=10**-6,
                 fibre_loss_init=0,fibre_loss_length=0):
        #print("BA_BB84_keygen")
        loc_BB84=BB84(num_bits=num_bits,fibre_len=fibre_len,
            fibre_loss_init=fibre_loss_init,fibre_loss_length=fibre_loss_init)
        # assign keys to A and B
        self.key_BB84_A=loc_BB84.key_A
        self.key_BB84_B=loc_BB84.key_B
        # trigger A's one way function
        self.node_A.ports["portC_A1"].tx_output('Alice')
        
        
        
    def B_send_GHZ(self,username):
        #print("B_send_GHZ called")
        if username.items[0]=='Alice':
            self.BChequeBook,tmpAChequeBook,tmpA2ChequeBook=Create_GHZ_triple_qubits_list(self.num_bits)
            tmpAChequeBook.extend(tmpA2ChequeBook)
            self.node_B.ports["portQ_B1"].tx_output(tmpAChequeBook)
        else:
            print("User name ERROR!")
    
    
    
    def A_req_USN(self,tmpAChequeBook):
        self.A1ChequeBook=tmpAChequeBook.items[:self.num_bits]
        self.A2ChequeBook=tmpAChequeBook.items[self.num_bits:]
        self.node_A.ports["portC_A2"].tx_output("USN_req")
        
        
        
    
    def B_USN_send_A(self,req): 
        #print("B_USN_send_A called")
        self.B_saveUSN=UniqueSerialNumGen(self.num_bits)
        #print(self.B_saveUSN)
        self.node_B.ports["portC_B3"].tx_output(self.B_saveUSN)
        
        
        
        
    def A_owf_send_C(self,saveUSN):
        #print("A_owf_send_C called")
        self.A_saveUSN=saveUSN.items
        if len(self.A1ChequeBook) < self.num_bits:
            print("No more cheque! Aborting!")
            return 0
        else:
            for i in range(self.num_bits):
                
                # Alice write down the amound of money 
                res_owf_qubit=OneWayFunction('Alice',self.key_BB84_A
                    ,self.A_saveUSN[i],self.Money)
                
                # Alice performs Bell state measurement
                operate([res_owf_qubit, self.A1ChequeBook[i]], ops.CNOT)     
                H | res_owf_qubit

                mes_A1=ns.qubits.qubitapi.measure(self.A1ChequeBook[i]
                    ,observable=Z)

                mes_owf_qubit=ns.qubits.qubitapi.measure(res_owf_qubit
                    ,observable=Z)

                if mes_A1[0] == 0   and mes_owf_qubit[0] == 1:
                    Z | self.A2ChequeBook[i]
                if mes_A1[0] == 1   and mes_owf_qubit[0] == 0:
                    X | self.A2ChequeBook[i]
                if mes_A1[0] == 1   and mes_owf_qubit[0] == 1:
                    Y | self.A2ChequeBook[i]


        self.node_A.ports["portQ_A2"].tx_output(self.A2ChequeBook)


            
            
    
    # C receives qubits from A then wait and send to B 
    def C_rec_wait_send_B(self,chequeQList):
        #print("C_rec_wait_send called")
        chequeQList=chequeQList.items
        
        self.C_Qmemory.put(chequeQList) 
        
        # wait for some time before summit to bank
        #print("wait for ",self.C_delay, " ns")
        My_waitENVtype = EventType("WAIT_EVENT", "Wait for N nanoseconds")
        self._schedule_after(self.C_delay, My_waitENVtype) # self.delay
        self._wait_once(ns.EventHandler(self.CpopMem),entity=self
            ,event_type=My_waitENVtype) # can't add event_type
        
    
    # pop qubits from qMemory
    def CpopMem(self,event):
        #print("time: ",ns.sim_time())
        # pop out from qmem
        sening_qList=[]
        sening_qList=self.C_Qmemory.pop(list(np.arange(self.num_bits))) #pop all
        #print("poped from Qmem:",sening_qList)
        self.node_C.ports["portQ_C2"].tx_output(sening_qList)
        
        
        
    # a list of qubits is verified by bank
    def CB_Verify(self,chequeQList):
        #print("CB_Verify called")
        chequeQList=chequeQList.items
        
        # error correction
        for i in range(self.num_bits):
            H | self.BChequeBook[i]
            bob_measurement = ns.qubits.qubitapi.measure(self.BChequeBook[i]
                        ,observable=Z)
            #print("bob_measurement=",bob_measurement)                  
            if bob_measurement[0] == 1:
                Z | chequeQList[i]
                
        # to use a fake cheque or not
        test_var = 'n'  # input("Use a fake check? (y/n)")
        res_closeness=[]
        
        for i in range(self.num_bits):
            if test_var == 'n':
                owf_bank_state = OneWayFunction('Bob',self.key_BB84_B,self.B_saveUSN[i],self.Money)
            elif (test_var == 'y'):
                owf_bank_state = OneWayFunction('Bob',[1,0,1,1,0,1],0,20) # try any value
 
            res_closeness.append(SwapTest(owf_bank_state, chequeQList[i]))
        
        sum=0
        #print('Resulting array of SWAP test: ', res_closeness )
        for i in range(len(res_closeness)):
            if int(res_closeness[i][0])==0:
                sum+=res_closeness[i][1]
            else:
                sum+=1-res_closeness[i][1]
                
                
        self.chequeCloseness=sum/len(res_closeness)
        #print(self.chequeCloseness)
        
        if self.chequeCloseness>self.Threshold:
            # pass
            self.Varify=True
            #print('Verified! Cheque Accepted!')
        else:
            # fail
            self.Varify=False
            #print('FAILED! ABORTING!' )
    
    
    
    
    # base function ========================================================
    
    def __init__(self, num_bits=8,fibre_len_AC=10**-6
                 ,fibre_len_AB=10**-6,fibre_len_BC=10**-6
                ,Money=1234,depolar_rate=0,time_independent=False, 
                 Threshold=0.99,C_delay=0): 
        super().__init__()
        self.node_A = Node("A",ID=0,port_names=["portQ_A1","portC_A1","portC_A2","portQ_A2","portC_A3"])
        self.node_B = Node("B",ID=1,port_names=["portQ_B1","portC_B1","portC_B2","portQ_B2","portC_B3"])
        self.node_C = Node("C",ID=2,port_names=["portQ_C1","portQ_C2"])
        
        self.C_Qmemory = QuantumMemory("C_QMemory", num_bits,ID=0, 
            memory_noise_models=DepolarNoiseModel(depolar_rate=depolar_rate,
            time_independent=time_independent))  #normally time_independent is False unless using probability
        self.C_delay = C_delay
        
        self.MyQfiber_AC = None
        self.MyQfiber_AB = None
        self.MyQfiber_BC = None
        
        self.MyCfiber_AB = None
        self.MyCfiber_BA = None
        
        self.fiberLenth_AC = fibre_len_AC
        self.fiberLenth_AB = fibre_len_AB
        self.fiberLenth_BC = fibre_len_BC
        
        self.num_bits = num_bits
        self.A1ChequeBook = [] 
        self.A2ChequeBook = [] 
        self.BChequeBook = [] 

        self.A_saveUSN = [] #unique serial number stored in A
        self.B_saveUSN = [] #unique serial number stored in B
        
        self.key_BB84_A = []
        self.key_BB84_B = []
        
        self.Money = Money
        self.Threshold = Threshold #Threshold of closeness about whether to accept this cheque
        self.chequeCloseness = 0
        self.Varify = False # True = cheque accepted, False = cheque denied
        
        self.start()
        
        
    def stop(self):
        super().stop()
        self._running = False
        
        
    def is_connected():
        super().is_connected()
        pass
        
        
    def start(self):
        super().start()
        
        self.MyCfiber_AB=DirectConnection("CFibreConn_AB",
                ClassicalFibre("CFibre_A->B", length=self.fiberLenth_AB),
                ClassicalFibre("CFibre_B->A", length=self.fiberLenth_AB))
        self.MyCfiber_AB2=DirectConnection("CFibreConn_AB2",
                ClassicalFibre("CFibre_A->B", length=self.fiberLenth_AB),
                ClassicalFibre("CFibre_B->A", length=self.fiberLenth_AB))
        self.MyCfiber_BA=DirectConnection("CFibreConn_BA",
                ClassicalFibre("CFibre_A->B", length=self.fiberLenth_AB),
                ClassicalFibre("CFibre_B->A", length=self.fiberLenth_AB))
        
        
        self.MyQfiber_AC=QuantumFibre("MyQFibre_AC", length=self.fiberLenth_AC, 
            loss_model=None)#"default"
        self.MyQfiber_AB=QuantumFibre("MyQFibre_AB", length=self.fiberLenth_AB, 
            loss_model=None)
        self.MyQfiber_BC=QuantumFibre("MyQFibre_BC", length=self.fiberLenth_BC, 
            loss_model=None)
        

        self.node_A.connect_to(self.node_B, self.MyCfiber_AB,
            local_port_name="portC_A1", remote_port_name="portC_B1")
        self.node_B.connect_to(self.node_A, self.MyQfiber_AB,
            local_port_name="portQ_B1", remote_port_name="portQ_A1")
        self.node_A.connect_to(self.node_B, self.MyCfiber_AB2,
            local_port_name="portC_A2", remote_port_name="portC_B2")
        self.node_B.connect_to(self.node_A, self.MyCfiber_BA,
            local_port_name="portC_B3", remote_port_name="portC_A3")
        
        
        self.node_A.connect_to(self.node_C, self.MyQfiber_AC,
            local_port_name="portQ_A2", remote_port_name="portQ_C1")
        self.node_C.connect_to(self.node_B, self.MyQfiber_BC,
            local_port_name="portQ_C2", remote_port_name="portQ_B2")
        
        
        
        # set callback functions===================================================
        self.node_B.ports["portC_B1"].bind_input_handler(self.B_send_GHZ)
        self.node_A.ports["portQ_A1"].bind_input_handler(self.A_req_USN)
        self.node_B.ports["portC_B2"].bind_input_handler(self.B_USN_send_A)
        self.node_A.ports["portC_A3"].bind_input_handler(self.A_owf_send_C) 
        self.node_C.ports["portQ_C1"].bind_input_handler(self.C_rec_wait_send_B)
        self.node_B.ports["portQ_B2"].bind_input_handler(self.CB_Verify)
        
        # Start by BB84
        self.BA_BB84_keygen()
        
        


# In[4]:


def sim(run_times=1,delay=0,depolar_rate=0):
    closeness_List=[]
    for i in range(run_times):
        ns.sim_reset()
        qc=QuantumCheque(num_bits=10,C_delay=delay,Money=110,depolar_rate=depolar_rate)
        ns.sim_run()
        #print(qc.chequeCloseness)
        closeness_List.append(qc.chequeCloseness)

    return sum(closeness_List)/len(closeness_List)

#sim(2,10**10)


# In[10]:


#===========================================plot======================================================
import matplotlib.pyplot as plt

def QC_plot():
    y_axis=[]
    x_axis=[]
    run_times=50
    num_bits=10
    min_delay=0
    max_delay=5*10**7 #86400s = a day
    
    depolar_rate=100

    
    #first line
    for i in range(min_delay,max_delay,5*10**6):    #i in ns
        x_axis.append(i)
        y_axis.append(sim(run_times,i,depolar_rate))
        
        
        
    plt.plot(x_axis, y_axis, 'go-',label='depolar_rate = 100')
    
    
    # second line
    x_axis.clear()
    y_axis.clear()
    depolar_rate=50
    for i in range(min_delay,max_delay,5*10**6):  
        x_axis.append(i)
        y_axis.append(sim(run_times,i,depolar_rate))
    
    plt.plot(x_axis, y_axis, 'bo-',label='depolar_rate = 50')
    

        
    plt.ylabel('average cheque closeness')
    plt.xlabel('time wait in C (ns)')

    
    #plt.xscale('log')
    plt.legend()
    plt.savefig('plot.png')
    plt.show()

    

QC_plot()

