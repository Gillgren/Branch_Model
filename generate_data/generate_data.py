import numpy as np
import pandas as pd
import random

def generate_dataset(n_data):
    values = []
    for k in range(n_data):
        a = random.uniform(0, 1)
        x = random.uniform(0, 1)
        b = random.uniform(0, 1)
        y = ((a)*np.sin(10*x) + b)
        values.append([a,x,b,y])
    return values

values = generate_dataset(10000)

df = pd.DataFrame(values)
df.to_csv("data.csv", index=False)


