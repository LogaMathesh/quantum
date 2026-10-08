from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

qc = QuantumCircuit(1)
qc.h(0)

fig = qc.draw("mpl")
plt.show()