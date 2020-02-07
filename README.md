# NetSquid Protocols
Here are list of quantum protocol simulations made by NetSquid.



## Code Structure

This part means to help readers understand the code.

First of all, my code structure might not be the best one, I am open for discussion.
All protocols follows the same code structure shown below:

![NsProtocolCodeStructure](https://github.com/h-oll/netsquid-private/blob/master/NsProtocolCodeStructure_70.png)

The code is divided into blocks, each block defines functions related to their block definition.


1. **Import libraries**

  This block contains all import instructions, it may include other protocols as well.

2. **General Functions**

  Functions that can be reused by other protocols belong here.

3. **Quantum Processor/Program Definition**

  Quantum Processor and Quantum Programs are defined here.

4. **Local Protocol Party A**

  A customized local protocol representing one of the party.

5. **Local Protocol Party B**

  A customized local protocol representing another one of the party.


6. **Implementation and Hardware Configuration**

  Implementation function includes hardware configuration and protocol function calls.
  Hardware environment is configured/reconfigured right before every simulation.

  Implementation function is used to run the same protocol several times,
  then take average value of certain attribute of the protocol for statistical use.


7. **Plot Function** -opt

  A function to plot statistical results of above protocol.


# Quantum Protocol List
## Quantum Key Distribution
- BB84
- E91/Ekert/EPR
- E91/Ekert/EPR NS7.0 version


## Quantum Money
- Quantum Token
- Quantum Cheque

## Quantum Teleportation
- State Teleportation

## Others
- Quantum Memory
- NetSquid Library


## Contact
ChinTe Liao
liao.chinte@veriqloud.fr
