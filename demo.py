import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import tulip as tlp

# %% Plotting settings
tlp.aesthetics()

#%% Loading tulipe class
tl = tlp.tulipe()

# params : no_components , deriv, root_id, leaf_id, n_points n_init, no_clusters,
#          max_iter, spline_res, spline_degree, smoothing_factor, smoother, kernel


#%% Running pipeline
tl.run()

#%% Extracting data

raw_data = tl.raw_data # Raw data
preprocessed_data = tl.preprocessed_data  # Preprocessed data
reduced_data = tl.reduced_data  # PCA and metadata
trajectory = tl.trajectory  # Trajectory, pseudomixture, and metadata

#%% Sample plots
cMap = plt.get_cmap("coolwarm")

fnt = 20

dim1 = 0
dim2 = 1
dim3 = 2

# Create a 3D plot
fig = plt.figure(figsize = (5,5.5), dpi = 300)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(tl.reduced_data["PC"][:,dim1], tl.reduced_data["PC"][:,dim2], -tl.reduced_data["PC"][:,dim3],
           c= cMap(tl.trajectory['pseudoz']), s = 120,
           edgecolors='w', alpha = .01, linewidths= 2
           )


ax.scatter(tl.trajectory["spline_points"][:,dim1], tl.trajectory["spline_points"][:,dim2], -tl.trajectory["spline_points"][:,dim3],
           s = 10, c = 'k', alpha = 1, linewidths=1,
                )

azim = 125
elev = 25


ax.set_xlabel(f'$\eta_1$', fontsize = fnt, labelpad = -10)
ax.set_ylabel(f'$\eta_2$', fontsize = fnt, labelpad = -5)
ax.set_zlabel(f'$\eta_3$', fontsize = fnt, labelpad = -5)
ax.set_xticks(())
ax.set_yticks(())
ax.set_zticks(())
ax.azim = azim
ax.elev = elev
ax.set_proj_type('persp', focal_length=.2)

# plt.tight_layout()
plt.show()

#%% Computing some samples along the trajectory
samples = tl.get_samples(n_spectra = 11)

#%%

fig = plt.figure(figsize=(8,4), dpi = 300)
ax = fig.add_subplot(111, projection = '3d')
azim = -35
elev = 10
cMap = plt.get_cmap('coolwarm')

cnt = -1
id_E = np.zeros((11,))
id_T = np.zeros((11,))

for i in range(11):
    cnt += 1
    y = (samples['arclength'][i])*np.ones(raw_data['t_local'].shape)
    ax.plot(raw_data['t_local'][0:4000]*1e9, y[0:4000], tlp.max_scaling(samples["S_avg"][i,0:4000]), c = cMap(i/10), linewidth = .7)

ax.set_ylim((0,10)) # number of sigs
ax.set_xlim((0,400)) # time
ax.set_zlim((0,1)) # height
ax.set_box_aspect([1.5,3.5,1])

ax.grid(False)
ax.set_xticks((0,100,200,300,400))
ax.set_zticks((0,.5,1))
ax.set_ylim((0,0.9))
ax.azim = azim
ax.elev = elev
# ax.set_axis_off()
ax.set_proj_type('persp', focal_length=.3)
ax.tick_params(labelsize = fnt, direction = 'in')
ax.set_xlabel(r'$t$ (ns)', fontsize = fnt, labelpad = 10)
# ax.set_ylabel(r'Signals', fontsize = fnt, labelpad = 20)
ax.set_zlabel(r'$\hat{S}_*$', fontsize = fnt, labelpad = 6)

plt.tight_layout()

# plt.subplots_adjust(wspace = 0, hspace = 0)
plt.subplots_adjust(left = 0.1, right = .9, top = 1.2, bottom = -.1)

#%
plt.show()