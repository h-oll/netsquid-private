#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import netsquid as ns

from netsquid.nodes.node import Node
from netsquid.protocols import NodeProtocol

from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.components.models import  FibreDelayModel
from netsquid.components.models.qerrormodels import *
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components import QSource,Clock
from netsquid.components.qsource import SourceStatus

from netsquid.qubits.operators import Operator,create_rotation_op
from netsquid.qubits.qformalism import *
from random import randint


# In[16]:


# General functions/Quantum programs

# Z Rotation operators 
theta = np.pi/8
# 16 types of rotations
# R0
R22 =create_rotation_op(   theta/2, rotation_axis=(0, 0, 1))
R45 =create_rotation_op( 2*theta/2, rotation_axis=(0, 0, 1))
R67 =create_rotation_op( 3*theta/2, rotation_axis=(0, 0, 1))
R90 =create_rotation_op( 4*theta/2, rotation_axis=(0, 0, 1))
R112=create_rotation_op( 5*theta/2, rotation_axis=(0, 0, 1))
R135=create_rotation_op( 6*theta/2, rotation_axis=(0, 0, 1))
R157=create_rotation_op( 7*theta/2, rotation_axis=(0, 0, 1))
R180=create_rotation_op(   np.pi/2, rotation_axis=(0, 0, 1))
R202=create_rotation_op( 9*theta/2, rotation_axis=(0, 0, 1))
R225=create_rotation_op(10*theta/2, rotation_axis=(0, 0, 1))
R247=create_rotation_op(11*theta/2, rotation_axis=(0, 0, 1))
R270=create_rotation_op(12*theta/2, rotation_axis=(0, 0, 1))
R292=create_rotation_op(13*theta/2, rotation_axis=(0, 0, 1))
R315=create_rotation_op(14*theta/2, rotation_axis=(0, 0, 1))
R337=create_rotation_op(15*theta/2, rotation_axis=(0, 0, 1))

#============================================================

INSTR_R22 = IGate('Z Rotated 22.5',operator=R22)
INSTR_R45 = IGate('Z Rotated 45'  ,operator=R45)
INSTR_R67 = IGate('Z Rotated 67.5',operator=R67)
INSTR_R90 = IGate('Z Rotated 90'    ,operator=R90)
INSTR_R112 = IGate('Z Rotated 112.5',operator=R112)
INSTR_R135 = IGate('Z Rotated 135'  ,operator=R135)
INSTR_R157 = IGate('Z Rotated 157.5',operator=R157)
#------------------------------------------------------------
INSTR_R180 = IGate('Z Rotated 180  ',operator=R180)
INSTR_R202 = IGate('Z Rotated 202.5',operator=R202)
INSTR_R225 = IGate('Z Rotated 225  ',operator=R225)
INSTR_R247 = IGate('Z Rotated 247.5',operator=R247)
INSTR_R270 = IGate('Z Rotated 270  ',operator=R270)
INSTR_R292 = IGate('Z Rotated 292.5',operator=R292)
INSTR_R315 = IGate('Z Rotated 315  ',operator=R315)
INSTR_R337 = IGate('Z Rotated 337.5',operator=R337)
#============================================================
INSTR_Rv22 = IGate('Z Rotated -22.5',operator=R22.inv)
INSTR_Rv45 = IGate('Z Rotated -45'  ,operator=R45.inv)
INSTR_Rv67 = IGate('Z Rotated -67.5',operator=R67.inv)
INSTR_Rv90 = IGate('Z Rotated -90'    ,operator=R90.inv)
INSTR_Rv112 = IGate('Z Rotated -112.5',operator=R112.inv)
INSTR_Rv135 = IGate('Z Rotated -135'  ,operator=R135.inv)
INSTR_Rv157 = IGate('Z Rotated -157.5',operator=R157.inv)
#------------------------------------------------------------
INSTR_Rv180 = IGate('Z Rotated -180  ',operator=R180.inv)
INSTR_Rv202 = IGate('Z Rotated -202.5',operator=R202.inv)
INSTR_Rv225 = IGate('Z Rotated -225  ',operator=R225.inv)
INSTR_Rv247 = IGate('Z Rotated -247.5',operator=R247.inv)
INSTR_Rv270 = IGate('Z Rotated -270  ',operator=R270.inv)
INSTR_Rv292 = IGate('Z Rotated -292.5',operator=R292.inv)
INSTR_Rv315 = IGate('Z Rotated -315  ',operator=R315.inv)
INSTR_Rv337 = IGate('Z Rotated -337.5',operator=R337.inv)

#INSTR_Swap = ISwap()


# In[26]:


# General functions/Quantum programs 

class PrepareEPRpairs(QuantumProgram):
    
    def __init__(self,pairs=1):
        self.pairs=pairs
        super().__init__()
        
    def program(self):
        qList_idx=self.get_qubit_indices(2*self.pairs)
        # create multiEPR
        for i in range(2*self.pairs):
            if i%2==0:                           # List A case
                self.apply(INSTR_H, qList_idx[i])
            else:                                # List B case
                self.apply(INSTR_CNOT, [qList_idx[i-1], qList_idx[i]])
        yield self.run(parallel=False)

 

'''
Measure the qubits hold by this processor by basisList.
input:
    basisList:list of int(0/1): indecate measurement basis
'''

class QMeasure(QuantumProgram):
    def __init__(self,basisList):
        self.basisList=basisList
        super().__init__()

    def program(self):
        #print("in QMeasure")
        for i in range(0,len(self.basisList)):
            if self.basisList[int(i/2)] == 0:  # basisList 0:Z  , 1:X    
                self.apply(INSTR_MEASURE, 
                    qubit_indices=i, output_key=str(i),physical=True) 
            else:                              
                self.apply(INSTR_MEASURE_X, 
                    qubit_indices=i, output_key=str(i),physical=True)

        yield self.run(parallel=False)


'''
input:
    positionInx:int List : Index in Qmem to measure.
    angleInx:int List (each value from 0 to 15): Index indecating measurement angle along Z-axis. #
output:
'''
class AngleMeasure(QuantumProgram):
    def __init__(self,positionInx,angleInx):
        self.positionInx=positionInx
        self.angleInx=angleInx
        super().__init__()

    def program(self):
        #print("in AngleMeasure")
        #print("self.positionInx",self.positionInx)
        #print("self.angleInx",self.angleInx)
        for pos,angle in zip(self.positionInx,self.angleInx):
            if   angle == 1:
                self.apply(INSTR_R22,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv22,pos)
            elif angle == 2:
                self.apply(INSTR_R45,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv45,pos)
            elif angle == 3:
                self.apply(INSTR_R67,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv67,pos)
            elif angle== 4:
                self.apply(INSTR_R90,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv90,pos)
            elif angle== 5:
                self.apply(INSTR_R112,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv112,pos)
            elif angle== 6:
                self.apply(INSTR_R135,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv135,pos)
            elif angle== 7:
                self.apply(INSTR_R157,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv157,pos)
            elif angle== 8:
                self.apply(INSTR_R180,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv180,pos)
            elif angle== 9:
                self.apply(INSTR_R202,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv202,pos)
            elif angle== 10:
                self.apply(INSTR_R225,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv225,pos)
            elif angle== 11:
                self.apply(INSTR_R247,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv247,pos)
            elif angle== 12:
                self.apply(INSTR_R270,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv270,pos)
            elif angle== 13:
                self.apply(INSTR_R292,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv292,pos)
            elif angle== 14:
                self.apply(INSTR_R315,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv315,pos)
            elif angle== 15:
                self.apply(INSTR_R337,pos)
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
                self.apply(INSTR_Rv337,pos)
            else:  # angle== 0
                self.apply(INSTR_MEASURE_X,qubit_indices=pos, output_key=pos,physical=True)
        
        
        yield self.run(parallel=False)
        
'''
input:
    Pg: A quantum program (QuantumProgram)
output:
    resList: A list of outputs from the given quantum program, 
    also sorted by key.(list of int)
'''

def getPGoutput(Pg):
    resList=[]
    tempDict=Pg.output
    if "last" in tempDict:
        del tempDict["last"]

    # sort base on key
    newDict=sorted({int(k) : v for k, v in tempDict.items()}.items())
    
    #take value
    for k, v in newDict:
        resList.append(v[0])
        #print("k,v: ",k,v[0])
    return resList


'''
Swap the qubits hold by this processor by position.
input:
    position:list of int: indecate qubits to swap 

'''

class QSwap(QuantumProgram):
    def __init__(self,position):
        self.position=position
        super().__init__()
        if len(position)!=2:
            print("Error parameters in QSwap!")
        

    def program(self):
        #print("in QSwap ")
        self.apply(INSTR_SWAP, qubit_indices=self.position, physical=True)
        yield self.run(parallel=False)    

'''
Apply CZ in a Qmem.
input:
    position:(list of two. ex [0,1])position to apply CZ
'''
class QCZ(QuantumProgram):
    def __init__(self,position):
        self.position=position
        super().__init__()

    def program(self):
        #print("in QCZ ")
        self.apply(INSTR_CZ, qubit_indices=self.position, physical=True)
        yield self.run(parallel=False)
        
        


# In[27]:


# server protocol
class ProtocolServer(NodeProtocol):

    def GetPGoutput_m1m2(self,QG):
        #print("S GetPGoutput_m1m2")
        tmp=getPGoutput(QG)
        #print("tmp:",tmp)
        self.m1=tmp[0]
        self.m2=tmp[1]
    
    def __init__(self,node,processor,port_names=["portQS_1","portCS_1","portCS_2"],realRound=5):
        super().__init__()
        self.node=node
        self.processor=processor
        self.portNameQ1=port_names[0]
        self.portNameC1=port_names[1]
        self.portNameC2=port_names[2]
        self.sourceQList=[]
        self.port_output=[]
        self.realRound=realRound
        
        
        self.S_Source = QSource("S_source") 
        self.S_Source.ports["qout0"].bind_output_handler(self.store_output_from_port)
        self.S_Source.status = SourceStatus.EXTERNAL
        
        self.C_delta1=None
        self.C_delta2=None
        self.m1=None
        self.m2=None
        
        
    def S_genQubits(self,num,freq=1e9):
        #generat qubits from source
        
        #set clock
        clock = Clock("clock", frequency=freq, max_ticks=num)
        try:
            clock.ports["cout"].connect(self.S_Source.ports["trigger"])
        
        except:
            print("already connected")
        
        clock.start()
        
        
    def store_output_from_port(self,message):
        self.port_output.append(message.items[0])
        if len(self.port_output)==4:
            #print("S store qubits:",self.port_output)
            self.processor.put(qubits=self.port_output)
            
            # do H CNOT operation
            # PrepareEPRpairs
            prepareEPRpairs=PrepareEPRpairs(2)
            
            
            self.processor.execute_program(
                prepareEPRpairs,qubit_mapping=[i for  i in range(0, 4)])
    
    
    def S_sendEPR(self):
        #print("S_sendEPR")
        payload=self.processor.pop([1,3]) # send the third one
        self.node.ports[self.portNameQ1].tx_output(payload)        
        

        
    def ProgramFail(self):
        print("S programe failed!!")
    
    
    def run(self):
        #print("server on")
        port = self.node.ports["portCS_1"]
        
        #receive classical from client
        yield self.await_port_input(port)
        rounds = port.rx_input().items
        #print("S received rounds:",rounds)
        
        # send half of an EPRpair to client
        
        # gen 2 qubits
        self.S_genQubits(4)
        # EPR pair formed when port received
        
        
        yield self.await_program(processor=self.processor)
        #print("S1 num_used_positions=",self.processor.num_used_positions)
        self.S_sendEPR()
        
        
        port = self.node.ports["portCS_1"]
        #receive qubits from client
        yield self.await_port_input(port)
        #print("S2 num_used_positions=",self.processor.num_used_positions)
        ack=port.rx_input().items[0]
        if ack!='ACK':
            print("ACK ERROR!")
        
        '''
        if tmp[0]=="ACK":
            #print("S ACK received start swaping")
        
        else:
            print(tmp[0])
            print("S ACK NOT received ERROR!!!")
        
        '''
        
        #Swap
        myQSwap=QSwap(position=[0,2])
        self.processor.execute_program(myQSwap,qubit_mapping=[0,1,2])
        
        
        #yield self.await_program(processor=self.processor)
        #print("S myQSwap finished")
        
        # send ACK
        
        self.node.ports["portCS_2"].tx_output("ACK2")
        
        # waiting for ACK3
        port = self.node.ports["portCS_1"]
        yield self.await_port_input(port)
        ack=port.rx_input().items[0]
        if ack!='ACK3':
            print("ACK3 ERROR!")
            
        #print("S received:",tmp)
        '''
        if tmp[0]=="ACK3":
            print("S ACK3 received start CNOT")
        else:
            print(tmp[0])
            print("S ACK3 NOT received ERROR!!!")
        '''
        
        
        myQCZ=QCZ(position=[0,2])
        self.processor.execute_program(myQCZ,qubit_mapping=[0,1,2])
        self.processor.set_program_fail_callback(self.ProgramFail,once=True)
        yield self.await_program(processor=self.processor)
        #print("S QCZ finished")
        
        #send ACK4
        self.node.ports["portCS_2"].tx_output("ACK4")
        
        
        # receiving theta1
        port = self.node.ports["portCS_1"]
        yield self.await_port_input(port)
        tmp=port.rx_input().items
        self.C_delta1=tmp[0]
        self.C_delta2=tmp[1]
        #print("S received self.C_delta1 :",self.C_delta2)
        
        # create customized measurement
        
        
        #yield self.await_program(processor=self.processor,await_fail=True)
        # S measure by 2 delta
        myAngleMeasure_m1m2=AngleMeasure([0,2],[self.C_delta1,self.C_delta2]) # first qubit
        self.processor.execute_program(myAngleMeasure_m1m2,qubit_mapping=[0,1,2])
        self.processor.set_program_done_callback(self.GetPGoutput_m1m2,myAngleMeasure_m1m2,once=True)
        self.processor.set_program_fail_callback(self.ProgramFail,once=True)
        
        yield self.await_program(processor=self.processor)
        #print("S self.m1 m2:",self.m1,self.m2)

        
        
        #send m1 m2
        self.node.ports["portCS_2"].tx_output([self.m1,self.m2])
        
        
        


# In[28]:


# client protocol
class ProtocolClient(NodeProtocol):
    
    def myGetPGoutput1(self,QG):
        if self.d == 2 :
            self.z2 = getPGoutput(QG)[0]
            #print("C self.z2=",self.z2)
        elif self.d == 1 :
            self.p2 = getPGoutput(QG)[0]
            #print("C self.p2=",self.p2)
        else:
            print("error")
            
    def myGetPGoutput2(self,QG):
        if self.d == 2 :
            self.p1 = getPGoutput(QG)[0]
            #print("C self.p1=",self.p1)
        elif self.d == 1 :
            self.z1 = getPGoutput(QG)[0]
            #print("C self.z1=",self.z1)
        else:
            print("error")
            
    
            
    def ProgramFail(self):
        print("C programe failed!!")
    
    
    def __init__(self,node,processor,rounds,port_names=["portQC_1","portCC_1","portCC_2"],maxRounds=10):
        super().__init__()
        self.node=node
        self.processor=processor
        self.portNameQ1=port_names[0]
        self.portNameC1=port_names[1]
        self.portNameC2=port_names[2]
        self.maxRounds=maxRounds
        self.rounds=rounds
        self.d=randint(1,2)
        self.z1=None
        self.z2=None
        self.p1=None
        self.p2=None
        
        self.theta1=None
        self.theta2=None
        self.r1=None
        self.r2=None
        
        self.delta1=None
        self.delta2=None
        
        self.m1=None
        self.m2=None
        
        self.verified=False
    
    def run(self):
        
        #print("client on")
        self.node.ports[self.portNameC1].tx_output(self.rounds)
        
        #receive qubits from client
        port = self.node.ports["portQC_1"]
        yield self.await_port_input(port)
        
        EPRpairs=port.rx_input().items
        #print("C received qubits:",EPRpairs)
        self.processor.put(EPRpairs)
        
        if self.d == 1 :
            #print("C case d=1")
            self.theta2=randint(0,7)
            self.r2=randint(0,1)
            # measure by theta2
            myAngleMeasure=AngleMeasure([0],[self.theta2]) # first qubit
            self.processor.execute_program(myAngleMeasure,qubit_mapping=[0])
            # assign p2
            self.processor.set_program_done_callback(self.myGetPGoutput1,myAngleMeasure,once=True)
            self.processor.set_program_fail_callback(self.ProgramFail,once=True)
            
        else:
            # measure the only qubit in Z basis
            #print("C case d=2")
            myQMeasure=QMeasure([0]) 
            self.processor.execute_program(myQMeasure,qubit_mapping=[0])
            # assign z2
            self.processor.set_program_done_callback(self.myGetPGoutput1,myQMeasure,once=True) 
            self.processor.set_program_fail_callback(self.ProgramFail,once=True)
            
            
        yield self.await_program(processor=self.processor)
        
        # send ACK
        #print("C sending  ACK")
        self.node.ports["portCC_1"].tx_output("ACK")

        #print("C waiting for ACK2")
        port = self.node.ports["portCC_2"]
        yield self.await_port_input(port)   
        ack = port.rx_input().items[0]
        #print("C received :",tmp)
        
        if ack!='ACK2':
            print("ACK2 ERROR!")
            
        
        #measure
        if self.d==1:
            #print("C case d=1")
            myQMeasure=QMeasure([0]) 
            self.processor.execute_program(myQMeasure,qubit_mapping=[0,1])
            self.processor.set_program_done_callback(self.myGetPGoutput2,myQMeasure,once=True) #not working
            self.processor.set_program_fail_callback(self.ProgramFail,once=True)
        else:
            #print("C case d=2")
            self.theta1=randint(0,7)
            self.r1=randint(0,1)
            # measure by theta1
            myAngleMeasure=AngleMeasure([1],[self.theta1]) # first qubit
            self.processor.execute_program(myAngleMeasure,qubit_mapping=[0,1])
            self.processor.set_program_done_callback(self.myGetPGoutput2,myAngleMeasure,once=True)
            self.processor.set_program_fail_callback(self.ProgramFail,once=True)
            
        
        yield self.await_program(processor=self.processor)
        
        # send ACK
        #print("C sending  ACK3")
        self.node.ports["portCC_1"].tx_output("ACK3")
        
        # wait ACK4
        #print("C waiting for ACK4")
        port = self.node.ports["portCC_2"]
        yield self.await_port_input(port)
        ack = port.rx_input().items[0]
        
        if ack!='ACK4':
            print("ACK4 ERROR!")
            #print("C received :",ack)
        
        # send theta1

        if self.d==1:
            self.delta1=randint(0,7)                      # scale x8 ; 1 = 22.5 degree
            self.delta2=self.theta2+(self.p2+self.r2)*8
            
        else:    
            self.delta1=self.theta1+(self.p1+self.r1)*8
            self.delta2=randint(0,7)
            
        
        self.delta1%=16
        self.delta2%=16
        #print("C delta1 delta2:",self.delta1,self.delta2)
        
        # send delta
        self.node.ports["portCC_1"].tx_output([self.delta1,self.delta2])
        
        
        # receive measurement result
        port = self.node.ports["portCC_2"]
        yield self.await_port_input(port)
        measRes = port.rx_input().items
        #print("C received measurement results: ",measRes)
        self.m1=measRes[0]
        self.m2=measRes[1]
        #print("C received measurement:",self.m1,self.m2)
        
        if self.d==1:
            #print("C d==1 case")
            if (self.z1+self.r2)%2== self.m2:
                #print("Varified!")
                self.verified=True
                #print("z1,r2,m2:",self.z1,self.r2,self.m2)
            else:
                #print("Failed!")
                self.verified=False
                #print("z1,r2,m2:",self.z1,self.r2,self.m2)
        else: # d==2
            #print("C d==2 case")
            if (self.z2+self.r1)%2== self.m1:
                #print("Varified!")
                self.verified=True
                #print("z2,r1,m1:",self.z2,self.r1,self.m1)
            else:
                #print("Failed!")
                self.verified=False
                #print("z2,r1,m1:",self.z2,self.r1,self.m1)


# In[29]:


# implementation & hardware configure
def run_UBQC_sim(runtimes=1,fibre_len=10**-9,processorNoiseModel=None,memNoiseMmodel=None
               ,loss_init=0.25,loss_len=0.2):
    
    resList=[]
    successCount=0
    
    for i in range(runtimes): 
        
        ns.sim_reset()

        # nodes====================================================================

        nodeServer = Node("Server", port_names=["portQS_1","portCS_1","portCS_2"])
        nodeClient = Node("Client"  , port_names=["portQC_1","portCC_1","portCC_2"])

        # processors===============================================================
        
        processorServer=QuantumProcessor("processorServer", num_positions=10,
            mem_noise_models=memNoiseMmodel, phys_instructions=[
            PhysicalInstruction(INSTR_X, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_H, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_CNOT,duration=1,q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_CZ,duration=1,q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R22, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R45, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R67, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R90, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R112, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R135, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R157, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R180, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R202, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R225, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R247, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R270, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R292, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R315, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R337, duration=1, parallel=True),
                
            PhysicalInstruction(INSTR_Rv22, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv45, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv67, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv90, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv112, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv135, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv157, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv180, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv202, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv225, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv247, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv270, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv292, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv315, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv337, duration=1, parallel=True),
            
            PhysicalInstruction(INSTR_SWAP, duration=1, parallel=True)])
        
        
        
        processorClient=QuantumProcessor("processorClient", num_positions=10,
            mem_noise_models=memNoiseMmodel, phys_instructions=[
            PhysicalInstruction(INSTR_X, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_Z, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_H, duration=1, q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_CNOT,duration=1,q_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_MEASURE, duration=1, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R22, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R45, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R67, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R90, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R112, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R135, duration=1, parallel=True),
            PhysicalInstruction(INSTR_R157, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv22, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv45, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv67, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv90, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv112, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv135, duration=1, parallel=True),
            PhysicalInstruction(INSTR_Rv157, duration=1, parallel=True)])


        # channels==================================================================
        
        
        MyQChannel=QuantumChannel("QChannel_S->C",delay=0
            ,length=fibre_len
            ,models={"myFibreLossModel": FibreLossModel(p_loss_init=loss_init, p_loss_length=loss_len, rng=None)})
        
        
        nodeServer.connect_to(nodeClient, MyQChannel,
            local_port_name =nodeServer.ports["portQS_1"].name,
            remote_port_name=nodeClient.ports["portQC_1"].name)
        

        MyCChannel = ClassicalChannel("CChannel_C->S",delay=0
            ,length=fibre_len)
        MyCChannel2= ClassicalChannel("CChannel_S->C",delay=0
            ,length=fibre_len)
        

        nodeClient.connect_to(nodeServer, MyCChannel,
                            local_port_name="portCC_1", remote_port_name="portCS_1")
        nodeServer.connect_to(nodeClient, MyCChannel2,
                            local_port_name="portCS_2", remote_port_name="portCC_2")


        protocolServer = ProtocolServer(nodeServer,processorServer)
        protocolClient = ProtocolClient(nodeClient,processorClient,1)
        protocolServer.start()
        protocolClient.start()
        #ns.logger.setLevel(1)
        stats = ns.sim_run()
        
        
        resList.append(protocolClient.verified)
        
    for i in resList:
        if i==True:
            successCount+=1

    return successCount/len(resList)


# In[30]:


# test
run_UBQC_sim(runtimes=3000,fibre_len=0
    ,processorNoiseModel=None,memNoiseMmodel=None,loss_init=0,loss_len=0)


# In[54]:


# plot function
import matplotlib.pyplot as plt

def UBQC_plot():
    y_axis=[]
    x_axis=[]
    run_times=10
    min_dis=0
    max_dis=80
    
    mymemNoiseMmodel=T1T2NoiseModel(T1=11, T2=10)
    myprocessorNoiseModel=DepolarNoiseModel(depolar_rate=200)

    # first curve
    for i in range(min_dis,max_dis,5):
        
        x_axis.append(i)
        successRate=run_UBQC_sim(runtimes=run_times,fibre_len=i
            ,processorNoiseModel=None   
            ,memNoiseMmodel=None)
        #myprocessorNoiseModel   # mymemNoiseMmodel,loss_init=0.25,loss_len=0.2
        y_axis.append(successRate)
        
        
        
    plt.plot(x_axis, y_axis, 'go-',label='default fibre')
    
    plt.title('UBQC')
    plt.ylabel('verified rate')
    plt.xlabel('fibre length (km)')
    
    
    plt.legend()
    plt.savefig('plot4.png')
    plt.show()



UBQC_plot()


# In[ ]:




