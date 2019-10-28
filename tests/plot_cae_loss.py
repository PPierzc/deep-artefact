import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

loss, epochs = np.load('../data/cae_loss.npy')

plt.figure(figsize=(10, 10))
sns.set_style('white')
sns.set_context('talk')
plt.plot(epochs, loss, 'k')
plt.xlabel('Epoch Number')
plt.ylabel('Reconstruction Loss $L_r$')
plt.title('Training with $\gamma = 0$')
plt.show()
