# Quantum Token Protocol

## Description
 The protocol provides a way for issuing an unforgeable token which can only be verified by issuer or other authorized party.


## Workflow
 1. Node B prepares N random qubits and record the states.
 2. Node B send the qubits to node A.
 3. Node A wait for T seconds.
 4. Node A send a challange request to node B.
 5. Node B replies a challange for Node A to solve.
 6. Node A measures its qubits according to challange.
 7. Node A send the measurement result to node B.
 8. Node B approves the answer if the correctness .


## Protocol parameters

- num_bits    : Number of qubits, higher value means higher security but higher cost in terms of qubits management.
- fiberLenth  : [km] Fiber length between two Nodes, long fiber cause more noise.
- depolar_rate: [Hz]/[prob] Parameter of deplar noise model, this will be a probability if timeIND is true.
- timeIND     : If ture, depolar_rate will be a probability. Otherwise in Hz.
- threshold   : The threshold that token issuer sets. Value can only be 0 to 1. Higher value means lower tolerancy and higher security, but less verifying successful rate.
- waitTime    : [sec] The time between node A receiving the qubits and sending challange request.
