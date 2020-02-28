# QLine sub-protocol
Author: ChinTe LIAO (liao.chinte@veriqloud.fr)

## Code Structure
![QLineSubProtocolCodeStructure](https://github.com/h-oll/netsquid-private/blob/master/QLine/QLine.png)

## Function
The QLine protocol generates a symmetric key between node A and B.

Qubit sending direction:
(D)------>(A)------>(B)------>(C)

## To Do
- Add more physical feature
- Modularization
- Increase extensibility

## Status
- The QLine protocol is working, and able to generate key with length (input_number/2) in average between node A and B.
- Noise/Loss models are applied.
- It is able to calculate total qubit loss, or key rate with given hardware configuration.
- Not able to generate key pairs for all 4 nodes. Forwarding functions are needed.
- Sub-QLine code is done as an atomic component.(generate key pair between two nodes)
