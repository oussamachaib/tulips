import numpy as np
import matplotlib.pyplot as plt
import time
import tulip
import pickle

# %% Plotting settings
plt.style.use('default')
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

#%% Load and preprocess
start_time = time.time()

raw_data = tulip.load_data(dataset="data_diffusion_raw",freqflag=False)
preprocessed_data = tulip.preprocess(raw_data)
reduced_data = tulip.run_pca(preprocessed_data, deriv=True, no_components=50)

#%% Main trajectory
trajectory = fn.get_trajectory(reduced_data["PC"], no_clusters = 9, n_init = 100, max_iter = 1000,
                            smoothing_factor= 0.001, spline_degree = 2, smoother = 'rbf', kernel = 'cubic', leaf_id=0)

end_time = np.round(time.time() - start_time, 2)
print('[Total computational time : ' + str(end_time) + ' s]')

#%% Loading features
with open('pickle_saves/features.pkl', 'rb') as file:
    features = pickle.load(file)

#%%
def avg(data, window_size):
    pad_size = window_size // 2
    padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
    smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='same')
    return smoothed_data[pad_size:-pad_size]

#%% Creating placeholders

f = features["frequency"]
r = avg(features["ratio"], 5)
d = avg(features["decay_time"], 5)
z = np.linspace(0,1,len(f))#samples["arclength"]
zs = z**2.65
spm = (trajectory['pseudoz'])**2.65

# Correcting frequency
fe = f.copy()
fe[(zs>=.24) & (zs<.70)] *= 2

#%% Mechanism probability
n = np.ones((25000,))
n[spm<.24] = 2
n[(spm>=.24) & (spm<.70)] = 1
n[spm>=.7] = 2


#%% PC placeholders
pz = trajectory["pseudoz"]
pzs = trajectory["pseudoz"]**2.65

PC1 = reduced_data["PC"][:,0]
PC2 = reduced_data["PC"][:,1]
PC3 = reduced_data["PC"][:,2]

#%% Plotting
fnt = 20

# Create a 3D plot
fig = plt.figure(dpi = 300)
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(PC1, PC2, -PC3, c=pzs, cmap = "twilight", vmin = 0, vmax = 1, s = 60,
#            edgecolors='k', alpha = .1, linewidths=.2,
#             )

dim1 = 0
dim2 = 1
dim3 = 2
cMap = plt.get_cmap("coolwarm")
ax.scatter(trajectory["spline_points"][:,dim1], trajectory["spline_points"][:,dim2], -trajectory["spline_points"][:,dim3], s = 20,
                cmap = cMap(trajectory['arclength']), edgecolors='k', alpha = 1, linewidths=2,
                )
ax.scatter(trajectory["spline_points"][:,dim1], trajectory["spline_points"][:,dim2], -trajectory["spline_points"][:,dim3], s = 20,
                c = cMap(trajectory['arclength']), edgecolors='w', alpha = 1, linewidths=0,
                )

azim = 125
elev = 25

#Plot each signal as an inset at each PCA point
for i in (np.arange(len(PC1))[::200]):
    ax.text(PC1[i], PC2[i], -PC3[i], s = int(n[i]), color='w', backgroundcolor = 'k', fontsize = 8, alpha = 1)

for i in (np.arange(len(trajectory["spline_points"][:,dim1]))[::100]):
    ax.text(trajectory["spline_points"][i,dim1], trajectory["spline_points"][i,dim2], -trajectory["spline_points"][i,dim3]+1,
            s = int(n[i]), color='w', backgroundcolor = 'k', fontsize = 8, alpha = 1)


ax.set_xlabel(f'$\eta_1$', fontsize = fnt, labelpad = -10)
ax.set_ylabel(f'$\eta_2$', fontsize = fnt, labelpad = -5)
ax.set_zlabel(f'$\eta_3$', fontsize = fnt, labelpad = -5)
ax.set_xticks(())
ax.set_yticks(())
ax.set_zticks(())
ax.azim = azim
ax.elev = elev
ax.set_proj_type('persp', focal_length=.2)

#plt.tight_layout()
plt.show()
