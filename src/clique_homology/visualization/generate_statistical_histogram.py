import matplotlib.pyplot as plt
import numpy as np

# Your data
d2_null = np.array([
    6.64028041,  3.92775504,  1.082094  , 12.66635717,  6.27622214,
    0.22730746,  1.31221815,  0.65902204,  2.66655345,  2.08471187,
    1.45490651,  8.61224136,  6.41653394,  0.37363045,  0.920126  ,
    0.35291544,  0.6866356 ,  0.66289594,  3.15366129,  2.06473718,
    0.74642362,  1.21079048,  3.06559811,  0.42151381,  4.20336517,
    2.34248078,  2.67873995,  3.28409954,  1.18643196,  6.01452165,
    1.06465171,  0.63115072,  3.46627998,  4.87812025,  2.15193326,
    2.13751865, 13.03152068,  2.36057298,  3.24724786,  1.51678937,
    0.51330735,  2.11596767,  1.92295923,  1.10918574,  7.90196192,
    2.62922041,  0.59905362,  0.64986614,  1.48370522,  0.92725083,
    6.28829937,  0.58488381,  0.96239303,  0.72790302,  7.44230907,
    0.37486085,  1.65397909,  2.14454664,  3.32172681,  6.1526281 ,
    1.38205451,  1.69415031,  9.0136079 ,  6.12157934,  1.40848911,
    3.23152451,  2.53754858,  3.3414668 ,  1.04938021,  2.52854192,
    2.40730434,  1.21079048,  3.172862  ,  3.93568787,  2.44096985,
    2.10650723,  2.66885202,  4.86757792,  1.6349951 ,  6.40461107,
    1.06231265,  1.61672514,  2.02077559,  0.55632477,  4.23893834,
    1.04223718,  0.91835443,  1.05714528,  3.30262347,  3.83788628,
    1.3438876 ,  2.64716231,  1.01180289,  5.8471728 ,  2.36186829,
    5.37752385,  7.108538  ,  4.94848942,  1.09547683,  0.67366241
])

d2_obs = 1954.5423344525504

# Setup two side-by-side subplots with a shared y-axis
# Give the left plot (histogram) more width using gridspec
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 5), 
                               gridspec_kw={'width_ratios': [3, 1]})

# Adjust the space between the two plots to make it look like a single broken axis
fig.subplots_adjust(wspace=0.05)

# --- Left Plot (The Null Distribution) ---
ax1.hist(d2_null, bins=15, color='#8c92ac', edgecolor='black', alpha=0.8, 
         label='Null Distribution')
# Set x-limits to perfectly frame the null data
ax1.set_xlim(0, 14) 
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.tick_params(bottom=True, top=False, left=True, right=False)

# --- Right Plot (The C. elegans Observation) ---
# We draw the vertical line representing the observation
ax2.axvline(d2_obs, color='#d32f2f', linewidth=4, linestyle='--', 
            label=f'C. elegans (d ≈ {int(d2_obs)})')
# Set x-limits to frame the observation
ax2.set_xlim(d2_obs - 5, d2_obs + 5)
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(bottom=True, top=False, left=False, right=False)
# Ensure we only show the 1954 tick mark to keep it clean
ax2.set_xticks([1954]) 

# --- Drawing the Break Marks (the diagonal slashes) ---
d = 0.02 # size of the diagonal lines
kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False, linewidth=1.5)
# Slashes on the left axis
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
# Slashes on the right axis
kwargs.update(transform=ax2.transAxes)  
ax2.plot((-d * 3, +d * 3), (-d, +d), **kwargs) # Multiply d by 3 to match the 3:1 width ratio

# --- Labels and Legends ---
ax1.set_ylabel('Frequency', fontsize=14)
fig.suptitle('Mahalanobis Distance: Null Model vs. C. elegans', fontsize=16, fontweight='bold', y=0.95)
fig.text(0.5, 0.02, 'Distance from Mean', ha='center', fontsize=14)

# Place legends manually to sit nicely in their respective subplots
ax1.legend(loc='upper right', fontsize=11)
ax2.legend(loc='upper center', fontsize=11)

# Save the figure
plt.savefig('src/clique_homology/visualization/plots/Slide12_Visualization_BrokenAxis.png', dpi=300, bbox_inches='tight')