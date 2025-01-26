import anfis
import membership  # import membershipfunction, mfDerivs
import numpy as np

# Load training set data (make sure the path to the file is correct)
ts = np.loadtxt("trainingSet.txt", usecols=[1, 2, 3])  # Update path if needed
X = ts[:, 0:2]
Y = ts[:, 2]

# Define the membership functions
mf = [
    [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}], ['gaussmf', {'mean': -4., 'sigma': 10.}], ['gaussmf', {'mean': -7., 'sigma': 7.}]],
    [['gaussmf', {'mean': 1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}], ['gaussmf', {'mean': -2., 'sigma': 10.}], ['gaussmf', {'mean': -10.5, 'sigma': 5.}]]
]

# Initialize the membership function and ANFIS model
mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)

# Train the ANFIS model using Hybrid Learning (Jang's method)
anf.trainHybridJangOffLine(epochs=20)

# Print the final consequent values and fitted values
print(f"Last consequent: {round(anf.consequents[-1][0], 6)}")
print(f"Second last consequent: {round(anf.consequents[-2][0], 6)}")
print(f"Fitted value at index 9: {round(anf.fittedValues[9][0], 6)}")

# Test the model's output against expected values
if (round(anf.consequents[-1][0], 6) == -5.275538 and
    round(anf.consequents[-2][0], 6) == -1.990703 and
    round(anf.fittedValues[9][0], 6) == 0.002249):
    print('Test is good')

# Plot errors and results
anf.plotErrors()
anf.plotResults()
