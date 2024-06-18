# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 21:15:20 2023

@author: Sophie Mancha
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# %%  FUNCTIONS


def calculate_msd(positions):
    num_steps, num_particles, num_dimensions = positions.shape
    msd = np.zeros(min(num_steps, 90))  # Limiting time steps 
    msd_std = np.zeros(min(num_steps,90))
    
    for t in range(1, min(num_steps,90)):
        displacement = positions[t:] - positions[:-t]
        squared_displacement = np.square(displacement).sum(axis=2).mean(axis=1)
        msd[t] = squared_displacement.mean()
        msd_std[t] = squared_displacement.std()

    return msd , msd_std




# %%

#Reading in data
# NP = No Pattern
# 1 = Tumor 1
# 2 = Tumor 2 


df_NP = pd.read_excel(r'file 1', usecols="C,E:G")
df_1 = pd.read_excel(r'file 2', usecols="C,E:G")
df_2 = pd.read_excel(r'file 3', usecols="C,E:G")


grouped_NP = df_NP.groupby('TRACK_ID')[['POSITION_X', 'POSITION_Y']]
grouped_1 = df_1.groupby('TRACK_ID')[['POSITION_X', 'POSITION_Y']]
grouped_2 = df_2.groupby('TRACK_ID')[['POSITION_X', 'POSITION_Y']]




positionsEX_NP = []
for particle_id, group in grouped_NP:
    positionsEX_NP.append(group.values)
    
positionsEX_1 = []
for particle_id, group in grouped_1:
    positionsEX_1.append(group.values)
    
positionsEX_2 = []
for particle_id, group in grouped_2:
    positionsEX_2.append(group.values)
    

positionsEX_NP = np.array(positionsEX_NP)
positionsEX_1 = np.array(positionsEX_1)
positionsEX_2 = np.array(positionsEX_2)



# Calculate MSD
msd_valuesEX_NP , msd_std_NP = calculate_msd(positionsEX_NP)
msd_valuesEX_1 , msd_std_1 = calculate_msd(positionsEX_1)
msd_valuesEX_2 , msd_std_2 = calculate_msd(positionsEX_2)



time_steps_NP = np.arange(len(msd_valuesEX_NP))*30
time_steps_1 = np.arange(len(msd_valuesEX_1))*30
time_steps_2 = np.arange(len(msd_valuesEX_2))*30



# %%

# Plot MSD 

plt.figure(figsize=(8, 6))


plt.loglog(time_steps_NP, msd_valuesEX_NP, 'k', markerfacecolor='none', markeredgecolor='k', label='MSD No Pattern', markersize=2, linewidth=2)
plt.loglog(time_steps_1, msd_valuesEX_1, color ='#FF1493',  markerfacecolor = 'none',  markeredgecolor = '#FF1493', label='MSD Tumor 1', markersize=2, linewidth=2)
plt.loglog(time_steps_2, msd_valuesEX_2, color='#008080', markerfacecolor = 'none',  markeredgecolor='#008080', label='MSD Tumor 2', markersize=2, linewidth=2)



# log log scale slope = 1
x= np.arange(1,51,1)
y = (10 *x ** 1) 
plt.loglog(x, y, '--k', label='Slope 1')


# # Specify the x-coordinate where you want to plot the vertical line
x_vertical_line = 960


# %% Calculate fit before x_vertical_line - NO PATTERN
# Extract the data points before x_vertical_line
x_range_before = 959  # Adjust the range as needed
x_fit_before_NP = time_steps_NP[(time_steps_NP >= x_vertical_line - x_range_before) & (time_steps_NP < x_vertical_line)]
y_fit_before_NP = msd_valuesEX_NP[(time_steps_NP >= x_vertical_line - x_range_before) & (time_steps_NP < x_vertical_line)]

# Perform linear regression before x_vertical_line
fit_before_NP = np.polyfit(np.log(x_fit_before_NP), np.log(y_fit_before_NP), 1)
fit_fn_before_NP = np.poly1d(fit_before_NP)


# # Print the slope of the linear fit before x_vertical_line
print(f"Slope of the No Pattern fit before x={x_vertical_line}: {fit_before_NP[0]:.4f}")


# Calculate fit after x_vertical_line NO PATTERN 
# Extract the data points after x_vertical_line
x_range_after = 2000  # Adjust the range as needed
x_fit_after_NP = time_steps_NP[(time_steps_NP > x_vertical_line) & (time_steps_NP <= x_vertical_line + x_range_after)]
y_fit_after_NP = msd_valuesEX_NP[(time_steps_NP > x_vertical_line) & (time_steps_NP <= x_vertical_line + x_range_after)]

# Perform linear regression after x_vertical_line
fit_after_NP = np.polyfit(np.log(x_fit_after_NP), np.log(y_fit_after_NP), 1)
fit_fn_after_NP = np.poly1d(fit_after_NP)

# Print the slope of the linear fit after x_vertical_line
print(f"Slope of the No Pattern fit after x={x_vertical_line}: {fit_after_NP[0]:.4f}")



# %% Calculate fit before x_vertical_line - TUMOR 1
x_fit_before_1 = time_steps_1[(time_steps_1 >= x_vertical_line - x_range_before) & (time_steps_1 < x_vertical_line)]
y_fit_before_1 = msd_valuesEX_1[(time_steps_1 >= x_vertical_line - x_range_before) & (time_steps_1 < x_vertical_line)]

# Perform linear regression before x_vertical_line
fit_before_1 = np.polyfit(np.log(x_fit_before_1), np.log(y_fit_before_1), 1)
fit_fn_before_1 = np.poly1d(fit_before_1)


# Print the slope of the linear fit before x_vertical_line
print(f"Slope of the Tumor 1 fit before x={x_vertical_line}: {fit_before_1[0]:.4f}")

# Calculate fit after x_vertical_line - TUMOR 1 
x_fit_after_1 = time_steps_1[(time_steps_1 > x_vertical_line) & (time_steps_1 <= x_vertical_line + x_range_after)]
y_fit_after_1 = msd_valuesEX_1[(time_steps_1 > x_vertical_line) & (time_steps_1 <= x_vertical_line + x_range_after)]

# Perform linear regression after x_vertical_line
fit_after_1 = np.polyfit(np.log(x_fit_after_1), np.log(y_fit_after_1), 1)
fit_fn_after_1 = np.poly1d(fit_after_1)


# Print the slope of the linear fit after x_vertical_line
print(f"Slope of the Tumor 1 fit after x={x_vertical_line}: {fit_after_1[0]:.4f}")


# %% Calculate fit before x_vertical_line - TUMOR 2
x_fit_before_2 = time_steps_2[(time_steps_2 >= x_vertical_line - x_range_before) & (time_steps_2 < x_vertical_line)]
y_fit_before_2 = msd_valuesEX_2[(time_steps_2 >= x_vertical_line - x_range_before) & (time_steps_2 < x_vertical_line)]

# # Perform linear regression before x_vertical_line
fit_before_2 = np.polyfit(np.log(x_fit_before_2), np.log(y_fit_before_2), 1)
fit_fn_before_2 = np.poly1d(fit_before_2)


# # Print the slope of the linear fit before x_vertical_line
print(f"Slope of the Tumor 2 fit before x={x_vertical_line}: {fit_before_2[0]:.4f}")


# # Calculate fit after x_vertical_line - TUMOR 2 

x_fit_after_2 = time_steps_2[(time_steps_2 > x_vertical_line) & (time_steps_2 <= x_vertical_line + x_range_after)]
y_fit_after_2 = msd_valuesEX_2[(time_steps_2 > x_vertical_line) & (time_steps_2 <= x_vertical_line + x_range_after)]

# # Perform linear regression after x_vertical_line
fit_after_2 = np.polyfit(np.log(x_fit_after_2), np.log(y_fit_after_2), 1)
fit_fn_after_2 = np.poly1d(fit_after_2)

# Print the slope of the linear fit after x_vertical_line
print(f"Slope of the Tumor 2 fit after x={x_vertical_line}: {fit_after_2[0]:.4f}")


#%% 


plt.legend()
plt.grid(True)
plt.xlabel('Time (min)')
plt.ylabel('MSD')
plt.title('Mean Squared Displacement of Particles')
plt.xlim(30, 9000)  # Set x-axis limits 
plt.ylim(30, 10000)  # Set y-axis limits 
plt.show()





