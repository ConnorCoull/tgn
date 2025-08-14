file = "d:\\SWAT\\network\\2015-12-28\\2015-12-28_113021_98.log.part12_sorted.csv"

import pandas as pd
import numpy as np
import os

df = pd.read_csv(file)

# print ts where is_attack is 1
for i, row in df.iterrows():
    print("Raw time at", i, ":", repr(df.at[i, "time"]))
    print("Length of time string at", i, ":", len(df.at[i, "time"]))
