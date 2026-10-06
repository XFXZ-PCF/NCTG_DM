# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:39:59 2025

@author: uqpfan
#this script is used to do a amplidtude calcluation.
"""
#import torch
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.patches as patches
import datetime, time
import pandas as pd
import tqdm
from tqdm import trange
import numpy as np
import os
import sys
import math
import shutil
import matplotlib.pyplot as plt
import pywt
from scipy.signal import cwt, morlet2
from WT_util import compute_cwt_with_mexican_hat, compute_wavelet_energy

# In[1]
#specify the datasourece
number_sampleGENERATE = 500 #used to be 50 and increase to 500
#gen_data_path = r'D:\VT_GENERATOR\2025WEEK1013\TL_pre\tramsfer-learningFROMQWEN\500samplegenerate.npy' #this specify the generated trajectories.
#gen_data = np.load(gen_data_path)  # Shape: [N, T] 500*30
#as you use the padding operation, the oriiginal fo training is not eligible anymore
gen_data_path = './extend_gendata.xlsx'
gen_data = pd.read_excel(io=gen_data_path) #this is the soure for source speed values.
#it's columns are number.
# In[2]
#let me specific the table to be used and filled.
amplit_filepath = r'D:\VT_GENERATOR\2025WEEK1013\TL_pre\Generated_Oscillation_statistics.xlsx'
ampli_file = pd.read_excel(amplit_filepath)
# In[3]
#start to doing a match speed drop..
#you need enusre it is a no null data..in time..
maxID = 100

def time2column(t: float) -> int:
    t_rounded = round(t, 1)
    col_index = 21 + int(round(t_rounded / 0.1))   
    column = max(1, min(341, col_index)) #for secure;
    return column
chunkSIZE = min (maxID, ampli_file.shape[0])

ampli_file['amplitude'] = ampli_file['amplitude'].astype(float)
#WE ONLY process first 100 samples 
for i in trange(chunkSIZE):
    pass
    vis_st = ampli_file.loc[i,'timeD_st(s)']
    vis_D_END = ampli_file.loc[i,'timeD_end(s)']
    #note that their id are matched.
    speed_at_vis_st = gen_data[time2column(vis_st)].iloc[i] #row, and columns
    speed_at_vis_D_END = gen_data[time2column(vis_D_END)].iloc[i]
    amplitude = speed_at_vis_st - speed_at_vis_D_END #calculate amplitude,
    amplitude = round(amplitude,4)
    
    #fill the amplitude into the file
    ampli_file.loc[i,'amplitude'] = amplitude

    assert(amplitude>=0)
# In[4]
# now you can compare two study data; using visualisation.
#generated and empricial
Empical_data = pd.read_excel(r"D:\VT_GENERATOR\2025WEEK1013\TL_pre\Empi_Oscillation_statistics.xlsx",
                             usecols=['ID', 'amplitude','osc_duration']) .copy(deep=True)
gene_data = ampli_file[["ID","amplitude","osc_duration"]].iloc[0:chunkSIZE]
gene_data = gene_data.copy(deep = True)
Excluded_IDS = [5,12,14,27,35,41,45,77] 
#after wards the data  will be processed as 
filtered_gene_data = gene_data[~gene_data['ID'].isin(Excluded_IDS)].copy()
# Assume you already have: Empical_data, gene_data, Excluded_IDS

# Filter gene_data by excluding specified IDs
# In[5]
#visualization
import seaborn as sns

# ===============================
# 1. Filter out excluded IDs
# ===============================

# Add a label column for group identification 
Empical_data['Group'] = 'Empirical'
filtered_gene_data['Group'] = 'Generated'

# Combine both datasets for joint visualization
combined_df = pd.concat([Empical_data, filtered_gene_data], ignore_index=True)

# ===============================
# 2. Compute descriptive statistics
# ===============================
stats = combined_df.groupby('Group')[['amplitude', 'osc_duration']].agg(
    ['min', 'max', 'median', 'mean']
).round(4)

print("=== Descriptive Statistics ===")
print(stats)

# ===============================
# 3. Visualization settings
# ===============================
sns.set(style="whitegrid", font_scale=1.2)
palette = {"Empirical": "#1f77b4", "Generated": "#ff7f0e"}

# ===============================
# 4. Boxplot + Violin plot
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Boxplot for amplitude ---
sns.boxplot(
    data=combined_df,
    x="Group", y="amplitude",
    palette=palette, ax=axes[0, 0]
)
axes[0, 0].set_title("Amplitude Distribution - Boxplot", fontsize = 16)
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("Amplitude (m/s)", fontsize = 12)

# --- Violin plot for amplitude ---
sns.violinplot(
    data=combined_df,
    x="Group", y="amplitude",
    palette=palette, ax=axes[0, 1]
)
#axes[0, 1].set_title("Amplitude Distribution - Violin Plot", fontsize = 16)
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("Amplitude (m/s)", fontsize = 18)

# --- Boxplot for osc_duration ---------------------------------------
sns.boxplot(
    data=combined_df,
    x="Group", y="osc_duration",
    palette=palette, ax=axes[1, 0]
)
axes[1, 0].set_title("Oscillation Duration - Boxplot", fontsize = 16)
axes[1, 0].set_xlabel("")
axes[1, 0].set_ylabel("Duration (s)", fontsize = 12)

# --- Violin plot for osc_duration ---
sns.violinplot(
    data=combined_df,
    x="Group", y="osc_duration",
    palette=palette, ax=axes[1, 1]
)
#axes[1, 1].set_title("Duration", fontsize = 16) #- Violin Plot
axes[1, 1].set_xlabel("")
axes[1, 1].set_ylabel("Duration (s)", fontsize = 18)

plt.tight_layout()
plt.savefig("comparison_box_violin.png", dpi=600, bbox_inches='tight')
plt.show()





# ===============================
# 5. Histogram comparison
# ===============================
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# --- Amplitude histogram ---
sns.histplot(
    data=combined_df, x="amplitude",
    hue="Group", kde=True,
    palette=palette, ax=axes[0], alpha=0.6
)
axes[0].set_title("Amplitude Distribution - Histogram", fontsize = 16)
axes[0].set_xlabel("Amplitude (m/s)", fontsize = 12)
axes[0].set_ylabel("Count", fontsize = 12)

# --- Duration histogram ---
sns.histplot(
    data=combined_df, x="osc_duration",
    hue="Group", kde=True,
    palette=palette, ax=axes[1], alpha=0.6
)
axes[1].set_title("Oscillation Duration Distribution - Histogram", fontsize = 16)
axes[1].set_xlabel("Duration (s)", fontsize = 12)
axes[1].set_ylabel("Count", fontsize = 12)

plt.tight_layout()
plt.savefig("comparison_histogram.png", dpi=600, bbox_inches='tight')
plt.show()

# ===============================
# 6. Joint distribution (Amplitude vs Duration)
# ===============================
g = sns.jointplot(
    data=combined_df,
    x="amplitude", y="osc_duration",
    hue="Group",
    kind="kde",  # Can also use "scatter"
    palette=palette,
    fill=True, alpha=0.5, height=8
)

#revise the xlables;

g.ax_joint.set_xlabel(
    "Amplitude (m/s)",
    fontsize=16
)

g.ax_joint.set_ylabel(
    "Duration (s)",
    fontsize=16
)
plt.suptitle("Joint Distribution: Amplitude vs. Oscillation Duration", y=1.02)
g.figure.savefig("joint_distribution.png", dpi=600, bbox_inches='tight')
plt.show()

