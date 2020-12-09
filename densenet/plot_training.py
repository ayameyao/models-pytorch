import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

file = pd.read_csv('train_log_DenseNet121.csv')
df = pd.DataFrame(file)
print(df)

#train
fig = plt.figure(figsize=(10,7))
ax = plt.axes()

plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss");

x = range(1, len(df['Train loss'].values) + 1)
ax.plot(x, df['Train loss'].values, '-g', label='train loss');
ax.plot(x, df['Test loss'].values, '-b', label='test loss');

plt.legend()
plt.show()

