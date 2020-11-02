# Universal Blind Quantum Computing

## Verifiable UBQC

### test subprotocol Steps
 1. C chooses d.
 2. S generates 4 qubits.
 3. S makes 2 EPR pairs.
 4. S sends two qubit(second and fourth) to C, sharing 2 EPR pairs.
 5. C if d=1, randomly chooses theda2 and r2. Measure the first qubit by theta2. Assign result to p2.
      if d=2, measures the first qubit in standard basis. Assign result to z2.
 6. C sends ACK to S.
 7. S swaps first/third qubit.
 8. S sends ACK2 to C.
 9. C if d=1, measures the first qubit in standard basis. Assign result to z1.
      if d=2, randomly chooses theda1 and r1. Measure the first qubit by theta2. Assign result to p1.
10. C sends ACK3 to S.
11. S does CNOT wiht first and third qubits.
12. S sends ACK4 to C.
13. C if d=1, randomly chooses delta1. Assign delta2=theta2+(p2+r2)*pi.
      if d=2, randomly chooses delta2. Assign delta2=theta2+(p2+r2)*pi.
14. C sends delta1 and delta2 to S.
15. S measures the first qubit by delta1. And measures the third qubit by delta2. Assign results to m1 and m2.
16. S sends m1 and m2 to C
17. C if d=1, and (z1+r2)%2=m2, than verified.
      if d=2, and (z2+r1)%2=m1, than verified.
      else Not verified.
