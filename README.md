# NetSquid Protocols
Here are list of quantum protocol simulations made by NetSquid.



## Code Structure

This part means to help readers understand the code.

First of all, my code structure might not be the best one, I am open to discuss about it.
All protocols follows the same code structure shown below:

![NsProtocolCodeStructure](https://github.com/h-oll/netsquid-private/blob/master/NsProtocolCodeStructure.png)

The code is divided into blocks, each block defines functions related to their block definition.


1. **Import libraries**

  This block contains all import instructions, it may include other protocols as well.

2. **General Functions**

  Functions that can be reused by other protocols belong here.

3. **Protocol Class**

  Every new protocol has to be declared as a class and inherit "Protocol" class definition in NetSquid.

  - **Operation functions**
  
    Functions that are used by the current protocol itself. They are customized to this particular protocol.

  - **Base functions**

    Due to the inheritance, there are four functions we need to declare:
    - **\_\_init__**

      Attributes of the protocol are initialized here. They are either physical attributes or data which needs to be stored between different stage of the exchange.

    - **stop**

      Not often used, but declaration is mandatory.

    - **is_connected**

      Not often used, but declaration is mandatory.

    - **start**

      Four implementations are defined here:
        - fiber configurations
        - port connections
        - define callback function
        - trigger of the starting function


4. **Implement function**

  A function used to call above protocol. It is used to run the same protocol several times, then get average value of certain attribute of the protocol for statistical use.

5. **Plot Function**

  A function to plot statistical results of above protocol.


# Quantum Protocol List
## Quantum Key Distribution
- BB84




## Quantum Money
- Quantum Token
- Quantum Check



## Contact
ChinTe Liao
liao.chinte@veriqloud.fr
