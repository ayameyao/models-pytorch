import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

file = pd.read_csv('train_log_DenseNet121.csv')
df = pd.DataFrame(file)
print(df)

fig = plt.figure(figsize=(10,7))
ax = plt.axes()

plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy");

x = range(1, len(df['Train accuracy'].values) + 1)
ax.plot(x, df['Train accuracy'].values, '-g', label='train accuracy');
ax.plot(x, df['Test accuracy'].values, '-b', label='test accuracy');

plt.legend()
plt.show()