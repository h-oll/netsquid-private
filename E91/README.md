# E91/Ekert/EPR Protocol
Author: ChinTe LIAO (liao.chinte@veriqloud.fr)

## Function

E91 protocol is one of Quantum Key Distribution that generate a key pair within two parties.


## To Do

- Parameter tuning.

## Status

E91 works on NS 0.10.
Noises applicable on quantum processor/memory/gates/source. Loss applicable in channels.


19/10/2020
- Add a version to plot key rate. Including a simple function for key length calibration.

07/09/2020
- Integrated Qsource.
- Add key pair generating functions for error correction test.

02/09/2020
- Replaced fibres with channels.
- Integreted old sendableQProcessor class to protocol functions.
- portName usage improved.

17/02/2020

- Bug fixed and be able to handle various qubit losses.

14/02/2020

- Quantum Processor interface applied when Bob measure.


07/02/2020

- Quantum Processor interface applied when Alice prepares qubits.
- Alice and Bob measure in normal interface.

01/04/2020

- Protocol interface upgrade to 8.0.
