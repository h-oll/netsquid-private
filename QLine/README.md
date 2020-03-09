# QLine sub-protocol
Author: ChinTe LIAO (liao.chinte@veriqloud.fr)

## Code Structure
![QLineSubProtocolCodeStructure](https://github.com/h-oll/netsquid-private/blob/master/QLine/QLine.png)

## Function
The protocol generates a symmetric key between each of the two nodes in QLine.

Qubit sending direction:
(A)------>(B)------>(C)------>(D)

## To Do
- Add more physical feature
- Modularization
- Increase extensibility

## Status
- The QLine protocol is able to generate key with length (input_number/2) in average between any given nodes in Qline.
- Noise/Loss models are able to apply.
- It is able to calculate total qubit loss, or key rate with given hardware configuration.
- Sub-QLine code is done as an atomic component.(generate key pair between two nodes)
