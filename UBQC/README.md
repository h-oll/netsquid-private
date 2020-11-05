# Universal Blind Quantum Computing

## Verifiable UBQC


### Test subprotocol variable ranges
- range A=n*pi/8, n=[0~7]
- range B=n*pi/8, n=[0~15]
- theta1: A
- theta2: A
- r1:[0,1]
- r2:[0,1]
- delta1: A or B
- delta2: A or B

**results:**
- p1:[0,1]
- p2:[0,1]
- z1:[0,1]
- z2:[0,1]

All angle measurements are along Z-axis

### Test subprotocol variable Steps

 1. C chooses d.
 2. S generates 4 qubits.(first paired with second, third paired with forth)
 3. S makes 2 EPR pairs.
 4. S sends two qubit(second and fourth) to C, sharing 2 EPR pairs.
 5. C if d=1, randomly chooses theta2 and r2, measure the first qubit by theta2, assign result to p2.
 
      if d=2, measures the first qubit in standard basis, assign result to z2.
 6. C sends ACK to S.
 7. S swaps first/third qubit.
 8. S sends ACK2 to C.
 9. C if d=1, measures the first qubit in standard basis, assign result to z1.
 
      if d=2, randomly chooses theta1 and r1, measure the first qubit by theta2. Assign result to p1.
10. C sends ACK3 to S.
11. S apply CZ with first and third qubits.
12. S sends ACK4 to C.
13. C if d=1, randomly chooses delta1, assign delta2=theta2+(p2+r2)*pi.

      if d=2, randomly chooses delta2, assign delta1=theta1+(p1+r1)*pi.
14. C sends delta1 and delta2 to S.
15. S measures the first qubit by delta1. And measures the third qubit by delta2, assign results to m1 and m2.
16. S sends m1 and m2 to C
17. C if d=1, and (z1+r2)%2=m2, than verified.

      if d=2, and (z2+r1)%2=m1, than verified.
      
      else Not verified.
