# QLine protocol
Author: ChinTe LIAO (liao.chinte@veriqloud.fr)

more information about [QLine](https://veriqloud.com/qline/)

## Code Structure
![QLineSubProtocolCodeStructure](https://github.com/h-oll/netsquid-private/blob/master/QLine/QLine.png)

Illustration below shows the Qubit flow every key bit

Qubit sending direction:
(A)------>(B)------>(C)------>(D)

According to roles of nodes, we have
(O)------>(I)------>(T)------>(D)
Indicating **Origin**,**Initial**,**Target** and **Destination**, where (I) can be (O)
as well as (T) being (D).

Origin : First Node of this QLine.
Initial: The initial node that start to generate a key pair.
Target : Target node that share a key pair with the initial node.
Destination: Last node of this QLine.

For a length fixed QLine, (O) and (D) are fixed, whereas (I) and (T) are chosen by user.

## Purpose
The protocol is able to generate a pair of symmetric key between each of the two nodes in QLine.



## To Do
- Add more physical parameters.

## Status
- The QLine functionality is completed.
It is able to generate a pair of key between any given nodes in Qline with Noise/Loss models applied.
