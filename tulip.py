import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.cluster import KMeans
import mat73
import scipy as sci
import networkx as nx
from scipy.spatial.distance import cdist
import pywt
import cv2
import os

#%% Primary functions
class tulipe():
    def __init__(self, no_components = 50, deriv = True, root_id = 2, leaf_id = -1, n_points = 1000, n_init = 100, no_clusters = 9, max_iter = 1000, spline_res = 1000,
                       spline_degree = 2, smoothing_factor = 0.001, smoother = 'spline', kernel = 'cubic'):
        self.dataset = "data_diffusion_raw"
        self.raw_data = {}
        self.preprocessed_data = {}
        self.no_components = no_components
        self.deriv = deriv
        self.root_id = root_id
        self.leaf_id = leaf_id
        self.n_points = n_points
        self.n_init = n_init
        self.no_clusters = no_clusters
        self.max_iter = max_iter
        self.spline_res = spline_res
        self.spline_degree = spline_degree
        self.smoothing_factor = smoothing_factor
        self.smoother = smoother
        self.kernel = kernel

    def run(self):
        print('Running tulip pipeline...')
        start_time = time.time()
        # Loading data
        self._load_data()
        # Preprocessing data
        self._preprocess()
        # Principal component analysis
        self._run_pca()
        # Learning trajectory
        self._get_trajectory()
        end_time = time.time() - start_time
        print('Elapsed time (tulip) : ' + str(end_time) + ' s')


    def _load_data(self):
        start_time = time.time()
        print('Loading dataset...')

        # Loading data
        data_path = "raw_data/"+self.dataset+".mat"
        data_files = mat73.loadmat(data_path)

        # Loading acquisition settings
        dt = data_files['dt']
        fs = data_files['fs']

        # Creating LIGS array
        t_local = data_files['t_local']
        t_local = t_local - t_local[0]
        t_local = t_local[0:6000]
        S = data_files['S_all']
        ishape = S.shape

        # Dictionary
        raw_data = {'S': S,
                't_local': t_local,
                'dt': dt,
                'fs': fs.item(),
                'ishape': ishape
                }

        # Time
        end_time = np.round(time.time() - start_time, 2)
        print('Elapsed time (Loading signals) : ' + str(end_time) + ' s')

        self.raw_data = raw_data

    def _preprocess(self):
        # Filtering settings
        b0, a0 = sci.signal.butter(4, Wn=140 * 1e6, fs = self.raw_data['fs'], btype='lowpass')

        # Preprocessing signals
        def _filter_center_scale():
            S_pre = max_scaling(centering(filtering(self.raw_data['S'], b0, a0)))
            G_pre = max_scaling(centering(np.gradient(filtering(self.raw_data['S'], b0, a0), axis=1)))
            return S_pre, G_pre

        S_pre, G_pre = _filter_center_scale()

        preprocessed_data = {'S_pre': S_pre,
                             'G_pre': G_pre
                            }

        self.preprocessed_data = preprocessed_data

    def _run_pca(self):
        start_time = time.time()
        print(f'Computing principal components...' )

        # PCA (first iteration)
        pca_model = PCA(n_components=self.no_components)
        if self.deriv:
            PC = pca_model.fit_transform(self.preprocessed_data['G_pre'])
        else:
            PC = pca_model.fit_transform(self.preprocessed_data['S_pre'])

        # Time
        end_time = np.round(time.time() - start_time, 2)
        print(f'Elapsed time (PCA : {self.no_components} components, {np.sum(pca_model.explained_variance_ratio_)*100:.0f}% total variance) : ' + str(end_time) + ' s')

        # Saving in dictionary
        reduced_data = {'PC': PC,
                        'pca_model': pca_model
                        }

        self.reduced_data = reduced_data

    def _get_trajectory(self):
        start_time = time.time()
        print('Trajectory inference...')

        # Root and lead nodes coordinates
        PC = self.reduced_data['PC']
        root_indices = np.linspace(int(self.root_id * self.n_points), int((self.root_id + 1) * self.n_points) - 1, self.n_points, endpoint=True, dtype=int)
        root_coords = np.median(PC[root_indices, :], axis=0)
        no_clusters_exclusive = self.no_clusters - 1

        if self.leaf_id >= 0:
            leaf_indices = np.linspace(int(self.leaf_id * self.n_points), int((self.leaf_id + 1) * self.n_points) - 1, self.n_points, endpoint=True, dtype=int)
            leaf_coords = np.median(PC[leaf_indices, :], axis=0)
            no_clusters_exclusive -= 1

        def _clustering():
            print(f'[*] Finding anchors via kmeans clustering (k = {self.no_clusters})...')
            if self.leaf_id < self.root_id:
                if self.leaf_id < 0:
                    PC_temp = np.concatenate((PC[0:root_indices[0], :], PC[(root_indices[-1] + 1):, :]))
                else:
                    PC_temp = np.concatenate((PC[0:(leaf_indices[0]), :], PC[(leaf_indices[-1] + 1):(root_indices[0]), :], PC[(root_indices[-1]+1):,:]))
            else:
                PC_temp = np.concatenate((PC[0:(root_indices[0]), :], PC[(root_indices[-1] + 1):(leaf_indices[0]), :], PC[(leaf_indices[-1]+1):,:]))

            # Setting anchoring points
            km = KMeans(n_clusters=no_clusters_exclusive, init='k-means++', n_init=self.n_init, max_iter=self.max_iter, verbose=0, random_state=None)
            km.fit(PC_temp)
            clabels_temp = km.labels_+1
            ccenters_temp = km.cluster_centers_

            if self.leaf_id < self.root_id:
                if self.leaf_id < 0:
                    clabels = np.concatenate((clabels_temp[0:root_indices[0]],
                                            np.zeros(self.n_points).astype(int),
                                            clabels_temp[root_indices[0]:],
                                            ))
                    ccenters = np.concatenate(([root_coords],
                                               ccenters_temp,
                                               ))
                else:
                    clabels = np.concatenate((clabels_temp[0:leaf_indices[0]],
                                          (self.no_clusters-1)*np.ones(self.n_points).astype(int),
                                          clabels_temp[leaf_indices[0]:root_indices[0]],
                                          np.zeros(self.n_points).astype(int),
                                          clabels_temp[root_indices[0]:],
                                          ))
                    ccenters = np.concatenate(([root_coords],
                                               ccenters_temp,
                                               [leaf_coords]
                                               ))
            else:
                clabels = np.concatenate((clabels_temp[0:root_indices[0]],
                                      np.zeros(self.n_points).astype(int),
                                      clabels_temp[root_indices[0]:leaf_indices[0]],
                                      (self.no_clusters-1)*np.ones(self.n_points).astype(int),
                                      clabels_temp[leaf_indices[0]:],
                                      ))
                ccenters = np.concatenate(([root_coords],
                                           ccenters_temp,
                                           [leaf_coords]
                                           ))

            return clabels, ccenters

        def _get_mst(clabels, ccenters):
            print('[*] Constructing minimum spanning tree...')
            # Euclidean distance matrix
            dmat = cdist(ccenters, ccenters, metric='euclidean')
            # Minimum spanning tree
            mst = sci.sparse.csgraph.minimum_spanning_tree(dmat)
            # MST in int format
            mst_matrix = mst.toarray().astype(int)
            # Create a graph from the MST dense array
            G = nx.from_numpy_array(mst_matrix)

            # Recursive depth-first search
            def _dfs_order(graph, start_node=0):  # ordered graph
                visited = []
                order = []

                def _dfs(node):  # depth-first search : travels across each branch in the graph
                    if node not in visited:
                        visited.append(node)
                        order.append(node)
                        for neighbor in graph.neighbors(node):
                            _dfs(neighbor)

                _dfs(start_node)
                return order

            # Get the order of points
            order = _dfs_order(G)
            # Save the ordered points
            ordered_points = ccenters[order,:]

            return ordered_points

        def _smoothing(ordered_points):
            # Smoothing trajectory using splines
            print('[*] Smoothing trajectory...')
            dimensionality = ordered_points.shape[1]

            # Create the curvilinear coordinate
            t = np.linspace(0, 1, ordered_points.shape[0]).reshape(-1, 1)

            # Fit interpolator to each dimension separately
            interpolators = []

            if self.smoother == 'spline':
                for i in range(dimensionality):
                    print(f'[**] Smoothing dimension by Univariate Splines ({i+1}/{dimensionality})...')
                    spl = sci.interpolate.UnivariateSpline(t, ordered_points[:, i].reshape(-1, 1), k=self.spline_degree, s=self.smoothing_factor)
                    interpolators.append(spl)
            elif self.smoother == 'rbf':
                for i in range(dimensionality):
                    print(f'[**] Smoothing dimension by RBF ({i+1}/{dimensionality})...')
                    spl = sci.interpolate.RBFInterpolator(t, ordered_points[:, i].reshape(-1, 1), kernel= self.kernel, smoothing=self.smoothing_factor)
                    interpolators.append(spl)
            else:
                print('Undefined smoothing method')

            # Computing spline coordinates
            t_fine = np.linspace(0, 1, self.spline_res).reshape(-1, 1)

            # Evaluate the interpolators on the fine grid
            spline_points = np.zeros((self.spline_res, dimensionality))
            for i, spl in enumerate(interpolators):
                spline_points[:, i] = spl(t_fine).flatten()

            # Computing arc length of trajectory
            arclength = max_scaling(
                np.insert(np.cumsum(np.sqrt(np.sum((np.diff(spline_points, axis=0) ** 2), axis=1))), 0, 0))

            return spline_points, arclength

        def _pseudoz(spline_points, arclength):
            # Assigning pseudo-Z
            print('[*] Assigning pseudomixture...')
            dist_matrix = abs(cdist(PC, spline_points, metric = 'mahalanobis'))
            closest_point = np.argmin(dist_matrix, axis=1)
            z = arclength[closest_point]
            return z


        # Calling functions
        cluster_labels, centroids = _clustering()
        ordered_points = _get_mst(cluster_labels, centroids)
        spline_points, arclength = _smoothing(ordered_points)
        z = _pseudoz(spline_points, arclength)

        # Dictionary
        trajectory = {'centroids': ordered_points,
                'spline_points': spline_points,
                'arclength': arclength,
                'pseudoz' : z
                }

        # Time
        end_time = np.round(time.time() - start_time, 2)
        print('Elapsed time (Trajectory inference) : ' + str(end_time) + ' s')

        self.trajectory = trajectory
    def get_samples(self, n_spectra = 100, n_neighbors = 100, constrained = True):
        # n_spectra : number of samples in total
        # n_neighbors : number of neighbors used for averaging (maximum, if there are fewer we use fewer)
        if constrained:
            dz = 1 / n_spectra # width of range where n_spectra should reside
            indices = -np.ones((n_spectra, n_neighbors), dtype=int)
            arclength = np.zeros((n_spectra,))

            for i in range(n_spectra):
                local_z = i / n_spectra
                range_of_signals = np.where((self.trajectory['pseudoz'] >= local_z) & (self.trajectory['pseudoz'] <= (local_z + dz)))[
                    0]
                if len(range_of_signals) > n_neighbors:
                    knn = NearestNeighbors(n_neighbors=int(n_neighbors), algorithm='auto').fit(
                        self.reduced_data["PC"][range_of_signals, :])
                    id_in_spline = np.argwhere(abs(self.trajectory['arclength'] - local_z + dz / 2) == np.min(abs(self.trajectory['arclength'] - local_z + dz / 2)))[0][0]
                    _, indices_closest_in_range = knn.kneighbors(self.trajectory['spline_points'][int(id_in_spline), :].reshape(1, -1))
                    indices[i, :] = range_of_signals[indices_closest_in_range]
                else:
                    indices[i, 0:int(len(range_of_signals))] = range_of_signals[0:int(len(range_of_signals))] # unranked by proximity

                arclength[i] += local_z

            # Average spectra at each pseudoz
            S_avg = np.zeros((n_spectra, self.preprocessed_data["S_pre"].shape[1]))
            S_std = np.zeros(S_avg.shape)
            G_avg = np.zeros(S_avg.shape)

            for i in range(len(arclength)):
                S_avg[i, :] = np.mean(self.preprocessed_data["S_pre"][indices[i, np.argwhere(indices[i, :] >= 0)], :], axis=0)
                S_std[i, :] = np.std(self.preprocessed_data["S_pre"][indices[i, np.argwhere(indices[i, :] >= 0)], :], axis=0)
                G_avg[i, :] = np.mean(self.preprocessed_data["G_pre"][indices[i, np.argwhere(indices[i, :] >= 0)], :], axis=0)

        else:
            # Getting pseudoz indices to sample from manifold
            id_spectra = np.linspace(0, len(self.trajectory['spline_points']), n_spectra, dtype=int, endpoint=False)
            # Finding nearest neigbors
            knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(self.reduced_data["PC"])
            _, indices = knn.kneighbors(self.trajectory['spline_points'][id_spectra, :])
            # Average spectra at each pseudoz
            S_avg = np.array([np.mean(self.preprocessed_data["S_pre"][indices[i], :], axis=0) for i in range(len(indices))])
            S_std = np.array([np.std(self.preprocessed_data["S_pre"][indices[i], :], axis=0) for i in range(len(indices))])
            G_avg = np.array([np.mean(self.preprocessed_data["G_pre"][indices[i], :], axis=0) for i in range(len(indices))])
            # Arclength
            arclength = self.trajectory['arclength'][id_spectra]

        # Dictionary of sampled signals
        samples = {'S_avg': S_avg,
                   'S_std': S_std,
                   'G_avg': G_avg,
                   'arclength': arclength,
                   }
        return samples

#%% Further analysis

def get_features(self, samples, t_array, ratio = True, frequency = False, decay = True, npoints_cwt = 200, verbose = True):
    start_time = time.time()
    print('Computing spectra features...')

    # Initializing placeholders
    n_spectra = samples["S_avg"].shape[0]

    def _get_ratio():
        print('[*] Computing IT/IE ratios...')
        R = np.zeros(n_spectra, )
        for i in range(n_spectra):
            sig0 = max_scaling(samples["S_avg"][i, :])
            sig2 = max_scaling(-np.gradient(max_scaling(np.gradient(sig0))))

            pks, _ = sci.signal.find_peaks(sig2, prominence=0.03)
            pks = pks[pks > 350]
            pks = pks[:2]
            del_t = 10
            id_E0 = pks[0] + np.argmax(sig0[pks[0] - del_t:pks[0] + del_t]) - del_t
            id_T0 = pks[1] + np.argmax(sig0[pks[1] - del_t:pks[1] + del_t]) - del_t
            R[i] = sig0[id_T0] / sig0[id_E0]
        return R

    def _get_frequency():
        F = np.zeros(n_spectra, )
        print('[*] Computing frequencies...')
        for i in range(n_spectra):
            if verbose:
                print(f'Signal [{i}/{n_spectra}]...')
            cwt = getcwt(samples["G_avg"][i, :][100:3000], t=t_array, plotting=False, verbose=False, npoints=npoints_cwt)
            F[i] = cwt["fpeak"]
        return F

    def _get_decay():
        dt = np.diff(t_array)[0]
        D = np.zeros((n_spectra,))
        CS = np.cumsum(samples["S_avg"], axis=1)
        CS /= CS[:, -1][:, np.newaxis]

        for i in range(0, n_spectra):
            D[i] = dt * (np.argwhere(CS[i, :] > 0.95)[0][0] - np.argwhere(CS[i, :] > 0.05)[0][0])
        return D

    if ratio:
        R = _get_ratio()
    else:
        R = []

    if frequency:
        F = _get_frequency()
    else:
        F = []

    if decay:
        D = _get_decay()
    else:
        D = []

    # Dictionary
    dicto = {'ratio': R,
             'frequency': F,
             'decay_time': D,
             }

    # Time
    end_time = np.round(time.time() - start_time, 2)
    print('Elapsed time (feature computation) : ' + str(end_time) + ' s')

    return dicto

def get_features_standalone(self, S, G, t_array, ratio = True, frequency = False, npoints_cwt = 200, verbose = False):
    start_time = time.time()
    print('Computing spectra features...')

    # Initializing placeholders
    n_spectra = S.shape[0]
    R = np.zeros(n_spectra,)
    F = np.zeros(n_spectra,)
    tau = np.zeros(n_spectra,)

    def _get_ratio():
        print('[*] Computing IT/IE ratios...')
        for i in range(n_spectra):
            sig0 = S[i, :]
            sig2 = max_scaling(-np.gradient(max_scaling(np.gradient(sig0))))

            pks, _ = sci.signal.find_peaks(sig2, prominence=0.03)
            pks = pks[pks > 300]
            pks = pks[:2]
            del_t = 10
            id_E = pks[0] + np.argmax(sig0[pks[0] - del_t:pks[0] + del_t]) - del_t
            id_T = pks[1] + np.argmax(sig0[pks[1] - del_t:pks[1] + del_t]) - del_t
            R[i] = S[i,id_T] / S[i,id_E]
        return R

    def _get_frequency():
        print('[*] Computing frequencies...')
        for i in range(n_spectra):
            if verbose:
                print(f'Spectrum {i}/{n_spectra}...')
            cwt = getcwt(G[i,100:2000], t=t_array, plotting=False, verbose=False, npoints=npoints_cwt)
            F[i] = cwt["fpeak"]
        return F

    if ratio:
        R = _get_ratio()
    else:
        R = []
    if frequency:
        F = _get_frequency()
    else:
        F = []

    # Dictionary
    dicto = {'R': R,
             'F': F,
             }

    # Time
    end_time = np.round(time.time() - start_time, 2)
    print('Elapsed time (feature computation) : ' + str(end_time) + ' s')

    return dicto

def get_synthetic_samples(self, n_spectra = 100):
    start_time = time.time()
    print('Computing synthetic spectra...')

    # Getting pseudoz indices to sample from manifold
    id_spectra = np.linspace(0, len(self.trajectory['spline_points']), n_spectra, dtype=int, endpoint=False)
    # Reconstructing synthetic spectra
    points_to_reconstruct = self.trajectory['spline_points'][id_spectra]
    S_avg = self.reduced_data['pca_model'].inverse_transform(points_to_reconstruct)

    # Time
    end_time = np.round(time.time() - start_time, 2)
    print('Elapsed time (synthetic spectra computation) : ' + str(end_time) + ' s')
    return S_avg


#%% Miscellaneous functions

def max_scaling(S):
    if S.ndim == 2:
        S_scaled = S / np.max(S, axis=1)[:, np.newaxis]
    else:
        S_scaled = S / np.max(S)

    return S_scaled

def filtering(S, b, a):
    S_filtered = np.reshape(sci.signal.lfilter(b, a, S.flatten()), S.shape)
    return S_filtered

def centering(S):
    S_centered = S - np.mean(S[:, -1000:], axis=1)[:, np.newaxis]
    return S_centered

def get_kde(X, x, y, b=1):
    kde = KernelDensity(bandwidth=b, algorithm='kd_tree', kernel='gaussian', metric='euclidean')
    kde.fit(X)

    xg, yg = np.meshgrid(x, y)

    density_raw = np.exp(kde.score_samples(np.column_stack((xg.flatten(), yg.flatten()))))
    density_norm = density_raw / max(density_raw)
    density_norm = np.reshape(density_norm, (len(xg), len(yg)))

    return density_norm

def movmean(sig,window_size = 10):
    # Create a 1D array of coefficients for the moving average filter
    kernel = np.ones(window_size)/window_size
    # Use numpy's convolve function to apply the moving average filter to the data
    filtered_data = np.convolve(sig, kernel, mode='same')
    return filtered_data

def getf(sig, fs, prom = 0.05, window_size = 1):
    # Get PSD frequencies
    sig = sig/max(sig)
    # Windowing the signal
    sig = sig*sci.signal.get_window('hann',len(sig), fftbins= True)
    # Zero-padding
    num_zeros = round(2**19 - len(sig)/2)
    pad_sig = np.pad(sig, (num_zeros, num_zeros), mode='constant', constant_values = 0)
    # Computing PSD using Welch's method
    frequencies_full, psd_full = sci.signal.welch(pad_sig, fs=fs, noverlap=0, nperseg=len(pad_sig))
    # Filtering PSD - unadvised
    psd_full = movmean(psd_full,window_size)
    # Converting frequencies to MHz
    frequencies = frequencies_full*1e-06
    frequencies = frequencies.flatten()
    # Keeping frequencies in the expected range
    psd = psd_full[np.where((frequencies>=20) & (frequencies<=150))[0]]
    frequencies = frequencies[np.where((frequencies>=20) & (frequencies<=150))[0]]
    # Normalized PSD
    psd = psd/max(psd)
    # Finding peaks
    peaks, properties = sci.signal.find_peaks(psd,prominence = prom)
    freq_peaks = [frequencies[j] for i, j in enumerate(peaks) if frequencies[j]>=20 and frequencies[j]<=150]
    psd_peaks = [psd[j] for i, j in enumerate(peaks) if frequencies[j]>=20 and frequencies[j]<=150]
    fwhm = sci.signal.peak_widths(psd, peaks, rel_height=0.5)
    print("Peaks occur at frequencies: " + str(np.round(freq_peaks,2)))
    return frequencies_full.flatten()*1e-06, psd_full/max(psd_full)#, freq_peaks, psd_peaks, fwhm

def getcwt(sig, t, dt = 1e-10, wavelet = 'cmor2.5-2.0', verbose = False, plotting = 0, npoints = 50, tmax = 600, magnitude = True):
    start_time = time.time()
    scales = np.linspace(40, 1000, npoints)  # Define scales for the wavelet
    coefficients, freqs = pywt.cwt(sig, scales, wavelet, sampling_period=dt)
    tg, fg = np.meshgrid(t * 1e9, freqs * 1e-6)

    if magnitude:
        coeffs = np.sqrt(coefficients.imag ** 2 + coefficients.real ** 2)
        coeffs = coeffs / max(coeffs.flatten())
        idx = np.where(coeffs == 1)
        fpeak = fg[idx].item()
        psd = coeffs[:, idx[1]]
    else:
        coeffs = [coefficients.real, coefficients.imag]
        magna = np.sqrt(coefficients.imag ** 2 + coefficients.real ** 2)
        magna /= max(magna.flatten())
        idx = np.where(magna == 1)
        fpeak = fg[idx].item()
        psd = magna[:, idx[1]]

    end_time = np.round(time.time() - start_time, 2)

    if verbose:
        print(fr'[CWT] Computation time : {end_time} seconds...')
        print(f'[CWT] Peak occurs at {fpeak[0]:.2f} MHz')
    if plotting:
        fig = plt.figure(figsize=(6, 3), dpi=200)
        ax = fig.add_subplot(121)
        ax.plot(t*1e9, sig, linewidth = 1)
        plt.ylim((-0.05, 1.05))
        plt.xlim((0, tmax))
        ax.set_xlabel('$t$ ($ns$)', fontsize = 20)
        ax.set_ylabel('$\hat{V}$ (-)', fontsize=20)
        plt.tick_params(axis='both', labelsize=20)
        ax = fig.add_subplot(122)
        ax.contourf(tg, fg, coeffs, levels=20, cmap='magma', norm=PowerNorm(gamma=2.0))
        ax.set_xlabel('$t$ ($ns$)', fontsize=20)
        ax.set_ylabel('$f$ ($\mathrm{MHz}$)', fontsize=20)
        plt.tick_params(axis='both', labelsize=20)
        plt.ylim((0, 140))
        plt.xlim((0, 200))
        plt.tight_layout()
        plt.show()

    cwt = {'tg': tg,
          'fg': fg,
          'coeffs': coeffs,
          'fpeak': fpeak,
          'psd': np.array(psd)
          }
    return cwt

def create_video(image_folder, fps = 20):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()

    img = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = img.shape

    video_name = 'ligs_video.mp4'
    save_folder = os.path.join(image_folder, video_name)
    output_video = cv2.VideoWriter(save_folder, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    cnt = 0
    for image in images:
        print(f'Iteration {cnt}')
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        output_video.write(frame)
        cnt += 1

    # Release the video writer and close the OpenCV window
    output_video.release()
    cv2.destroyAllWindows()

    print(f"Video saved as {video_name}")

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def aesthetics():
    print('Loading plotting settings...')
    plt.style.use('default')
    plt.rc('text', usetex=True)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

