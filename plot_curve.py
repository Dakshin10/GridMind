import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

rewards = np.load("outputs/train_rewards.npy")

def moving_avg(x, w=20):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.plot(moving_avg(rewards, 20))
plt.title("Learning Curve")
plt.savefig("plots/training_curve.png")
print("Plot saved successfully to plots/training_curve.png")
