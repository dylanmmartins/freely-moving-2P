


import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import skew
from scipy.ndimage import label
from tqdm import tqdm
import multiprocessing

import fm2p


def rate_map_mp(spike_rate, occupancy, ray_distances, ray_width, dist_bin_edges, dist_bin_size):

    N_angular_bins = int(360 / ray_width)
    N_distance_bins = len(dist_bin_edges) - 1

    rate_map = np.zeros((N_angular_bins, N_distance_bins))

    for f in range(len(spike_rate)):
        for a, ang in enumerate(np.arange(0, 360, ray_width)):
            for d, dist_bin_start in enumerate(dist_bin_edges[:-1]):
                dist_bin_end = dist_bin_start + dist_bin_size
                if (ray_distances[f, a] >= dist_bin_start) and (ray_distances[f, a] < dist_bin_end):
                    rate_map[a, d] += spike_rate[f]

    rate_map /= occupancy + 1e-6  # avoid division by zero

    return rate_map

def calc_MRL_mp(ratemap, ray_width, dist_bin_cents):

    N_angular_bins, N_distance_bins = ratemap.shape
    angs_rad = np.deg2rad(np.arange(0, 360, ray_width))

    # create a meshgrid of angles and distances
    angs_mesh, _ = np.meshgrid(angs_rad, dist_bin_cents, indexing='ij')

    # calculate the mean resultant vector
    mr = np.sum(ratemap * np.exp(1j * angs_mesh)) / (N_angular_bins * N_distance_bins)
    
    mean_resultant_length = np.abs(mr)

    return mean_resultant_length


def calc_shfl_mean_resultant_mp(spikes, useinds, occupancy, ray_distances, ray_width,
                             dist_bin_edges, dist_bin_size, dist_bin_cents, is_IEBC):

    N_frames = np.sum(useinds)

    shift_amount = np.random.randint(int(0.1*N_frames), int(0.9*N_frames))
    shifted_spikes = np.roll(spikes[useinds], shift_amount)
    shifted_ratemap = rate_map_mp(shifted_spikes, occupancy, occupancy, ray_distances, ray_width, dist_bin_edges, dist_bin_size)
    if is_IEBC:
        shifted_ratemap = np.max(shifted_ratemap) - shifted_ratemap + np.min(shifted_ratemap)
    shf_mrl = calc_MRL_mp(shifted_ratemap, ray_width, dist_bin_cents)

    return shf_mrl


class BoundaryTuning:
    def __init__(self, preprocessed_data):

        self.data = preprocessed_data

        self.ray_width = 3 # deg
        self.max_dist = 35 # cm
        self.dist_bin_size = 2.5 # cm
    
    def calc_allo_yaw(self):
        # calculate the head yaw in allocentric space from a head direction of 0 deg = facing rightwards
        self.head_ang = self.data['head_yaw_deg']

    def calc_allo_pupil(self):
        # calculate the pupil orientation in allocentric space from a head direction of 0 deg = facing rightwards
        pupil_angle = np.deg2rad(self.data['theta_interp'])
        self.pupil_ang = (self.head_ang + pupil_angle) % (2 * np.pi)

    def get_ray_distances(self, angle='head'):

        if angle == 'head':
            angle_trace = self.data['head_yaw_deg'].copy()
        elif angle == 'pupil':
            angle_trace = self.data['theta_interp'].copy()

        x_trace = self.data['head_x'].copy() / self.data['pxls2cm']
        y_trace = self.data['head_y'].copy() / self.data['pxls2cm']

        N_frames = np.sum(self.useinds)

        x_trace = x_trace[self.useinds]
        y_trace = y_trace[self.useinds]
        angle_trace = angle_trace[:-1][self.useinds]

        angle_trace = np.deg2rad(angle_trace)

        x1 = np.nanmean([
            self.data['arenaBL']['x'] / self.data['pxls2cm'],
            self.data['arenaTL']['x'] / self.data['pxls2cm']
        ])
        x2 = np.nanmean([
            self.data['arenaBR']['x'] / self.data['pxls2cm'],
            self.data['arenaTR']['x'] / self.data['pxls2cm']
        ])
        y1 = np.nanmean([
            self.data['arenaTL']['y'] / self.data['pxls2cm'],
            self.data['arenaTR']['y'] / self.data['pxls2cm']
        ])
        y2 = np.nanmean([
            self.data['arenaBL']['y'] / self.data['pxls2cm'],
            self.data['arenaBR']['y'] / self.data['pxls2cm']
        ])

        wall_entries = [
            [x1, y1, x2, y1],
            [x1, y1, x1, y2],
            [x2, y1, x2, y2],
            [x1, y2, x2, y2]
        ]

        rays_rad = np.zeros((N_frames, int(360 / self.ray_width)))
        for f in range(N_frames):
            for r in range(int(360 / self.ray_width)):
                rays_rad[f,r] = angle_trace[f] + np.deg2rad(r)
        
        self.ray_distances = np.zeros([
            np.size(rays_rad,0),
            int(360 / self.ray_width)
        ]) * np.nan

        for fr in tqdm(range(np.size(rays_rad,0))):

            ray_distances = []
        
            for ri in range(np.size(rays_rad,1)):
                intersections = []
                closest_walls = []
                
            
                ray_ang = rays_rad[fr, ri]

                ray_vec = np.vstack((
                    np.cos(ray_ang), # x
                    np.sin(ray_ang)  # y
                ))

                for wall in wall_entries:
                    start = np.array([wall[0], wall[1]])
                    end = np.array([wall[2], wall[3]])
                    vector = end - start

                    # calculate the determinant (if 0, lines are parallel, no intersection)
                    det = np.cross(vector, ray_vec.T)
                    if any(det == 0):
                        continue

                    # calculate the relative position of ray origin to wall start point
                    relative_pos = np.array([x_trace[fr], y_trace[fr]]) - start

                    # calculate how far along the wall line the intersection occurs
                    # (t = 0 -> wall.start; t = 1 -> wall.end)
                    t = np.cross(relative_pos, ray_vec.T) / det
                    # if t is not between 0 and 1, the intersection is outside the finite wall line
                    if np.all((t < 0) | (t > 1)):
                        continue  # skip

                    # after these checks are passed, calculate the intersection coordinates
                    intersection = (start + np.outer(t, vector)).flatten()

                    # check if the intersection is really in the direction of the ray
                    if np.all(np.dot(intersection - np.array([x_trace, y_trace]).T, ray_vec) < 0):
                        continue

                    intersections.append(intersection)
                    # calculate Euclidean distance from (x, y) to the intersection
                    distance = np.linalg.norm(intersection - np.array([x_trace[fr], y_trace[fr]]))
                    closest_walls.append(distance)

                if len(closest_walls)==0:
                    min_dist = np.nan
                    ray_distances.append(min_dist)
                else:
                    min_dist = np.nanmin(closest_walls) # distance of closest wall for that ray
                    ray_distances.append(min_dist) # append that distance to bin distance list
            
            self.ray_distances[fr,:] = np.array(ray_distances)

        # calculate distance bin edges
        self.dist_bin_edges = np.arange(0, self.max_dist + self.dist_bin_size, self.dist_bin_size)
        # calculate distance bin center positions
        self.dist_bin_cents = self.dist_bin_edges[:-1] + (self.dist_bin_size / 2)

        return self.ray_distances
    
    def calc_occupancy(self):
        N_angular_bins = int(360 / self.ray_width)
        N_distance_bins = len(self.dist_bin_edges) - 1
        self.occupancy = np.zeros((N_angular_bins, N_distance_bins))

        for d, dist_bin_start in enumerate(self.dist_bin_edges[:-1]):
            dist_bin_end = dist_bin_start + self.dist_bin_size
            # create a mask of where the distance falls within the current distance bin
            mask = (self.ray_distances >= dist_bin_start) & (self.ray_distances < dist_bin_end)
            # sum across frames to get occupancy for each angular bin
            self.occupancy[:, d] = np.sum(mask, axis=0)

        return self.occupancy
    
    def calc_rate_maps_mp(self):

        nCells = np.size(self.data['norm_spikes'], 0)
        N_angular_bins = int(360 / self.ray_width)
        N_distance_bins = len(self.dist_bin_edges) - 1

        n_proc = multiprocessing.cpu_count() - 1
        print('  -> Starting multiprocessing pool (n_workers: {}/{}).'.format(n_proc, multiprocessing.cpu_count()))
        num_proc_batches = int(np.ceil(nCells / n_proc))
        pbar = tqdm(total=num_proc_batches)

        def update_progress_bar(*args):
            pbar.update()

        spikes = self.data['norm_spikes'].copy()[:, self.useinds]

        pool = multiprocessing.Pool(processes=n_proc)

        mp_param_set = [
            pool.apply_async(
                rate_map_mp,
                args=(
                    spikes[cell_num,:],
                    self.occupancy,
                    self.ray_distances,
                    self.ray_width,
                    self.dist_bin_edges,
                    self.dist_bin_size
                ),
                callback=update_progress_bar
            ) for cell_num in range(nCells)
        ]
        mp_outputs = [result.get() for result in mp_param_set]

        self.rate_maps = np.zeros((nCells, N_angular_bins, N_distance_bins))

        for c, rmap in enumerate(mp_outputs):
            self.rate_maps[c,:,:] = rmap

        pbar.close()
        pool.close()

        return self.rate_maps
    
    def calc_rate_maps(self, use_mp=True):

        if use_mp is True:
            return self.calc_rate_maps_mp()

        spikes = self.data['norm_spikes'].copy()[:, self.useinds]

        N_cells = self.data['norm_spikes'].shape[0]
        N_angular_bins = int(360 / self.ray_width)
        N_distance_bins = len(self.dist_bin_edges) - 1

        self.rate_maps = np.zeros((N_cells, N_angular_bins, N_distance_bins))

        for c in tqdm(range(N_cells)):
            spike_rate = spikes[c,:]
            for f in range(len(spike_rate)):
                for a, ang in enumerate(np.arange(0, 360, self.ray_width)):
                    for d, dist_bin_start in enumerate(self.dist_bin_edges[:-1]):
                        dist_bin_end = dist_bin_start + self.dist_bin_size
                        if (self.ray_distances[f, a] >= dist_bin_start) and (self.ray_distances[f, a] < dist_bin_end):
                            self.rate_maps[c, a, d] += spike_rate[f]

            self.rate_maps[c,:,:] /= self.occupancy + 1e-6  # avoid division by zero

        return self.rate_maps
    
    def smooth_rate_maps(self):

        smoothed_rate_maps = self.rate_maps.copy()

        for c in range(self.rate_maps.shape[0]):
            # pad the rate map by concatenating three copies along the angular axis
            temp_padded = np.vstack((smoothed_rate_maps[c,:,:], smoothed_rate_maps[c,:,:], smoothed_rate_maps[c,:,:]))
            # smooth the padded rate map
            temp_smoothed = gaussian_filter(temp_padded, sigma=1)

            # slice the middle third to get the final smoothed rate map
            smoothed_rate_maps[c,:,:] = temp_smoothed[smoothed_rate_maps.shape[1]:2*smoothed_rate_maps.shape[1], :]

        self.smoothed_rate_maps = smoothed_rate_maps
        return self.smoothed_rate_maps
    
    def _invert_ratemap(self, ratemap):
        return np.max(ratemap) - ratemap + np.min(ratemap)
    
    def _measure_skewness(self, ratemap):
        skew_val = skew(ratemap.flatten())
        passes = skew_val < 0.
        return skew_val, passes
    
    def _calc_dispersion(self, ratemap):
        # convert spatial bins to cartesian coordinates in egocentric space
        N_angular_bins, N_distance_bins = ratemap.shape
        angs_rad = np.deg2rad(np.arange(0, 360, self.ray_width))
        dist_cents = self.dist_bin_cents

        x_coords = np.zeros((N_angular_bins, N_distance_bins))
        y_coords = np.zeros((N_angular_bins, N_distance_bins))

        # fill in the x and y coordinates
        for a in range(N_angular_bins):
            for d in range(N_distance_bins):
                x_coords[a, d] = dist_cents[d] * np.cos(angs_rad[a])
                y_coords[a, d] = dist_cents[d] * np.sin(angs_rad[a])

        # identify bins in the top 10% of the map
        threshold = np.percentile(ratemap, 90)
        top_bins = ratemap >= threshold

        # get the coordinates of the top bins
        top_x = x_coords[top_bins]
        top_y = y_coords[top_bins]

        if len(top_x) < 2:
            return np.inf  # not enough points to calculate dispersion

        # calculate the centroid of the top bins
        centroid_x = np.mean(top_x)
        centroid_y = np.mean(top_y)

        # calculate the dispersion as the mean distance from the centroid
        distances = np.sqrt((top_x - centroid_x)**2 + (top_y - centroid_y)**2)
        dispersion = np.mean(distances)

        return dispersion
    
    def _measure_dispursion(self, ratemap):

        normal_dispersion = self._calc_dispersion(ratemap)
        inv_ratemap = self._invert_ratemap(ratemap)
        inverted_dispersion = self._calc_dispersion(inv_ratemap)

        passes = inverted_dispersion < normal_dispersion

        return normal_dispersion, inverted_dispersion, passes
    
    def _calc_receptive_field_size(self, ratemap):
        # pad the angular axis with first and last columns to avoid edge effects
        padded_ratemap = np.vstack((ratemap[-1,:], ratemap, ratemap[0,:]))
        threshold = np.percentile(padded_ratemap, 50)
        binary_map = padded_ratemap >= threshold

        structure = np.ones((3,3))
        labeled_array, num_features = label(binary_map, structure=structure)
    
        if num_features == 0:
            return
        
        # remove the wrap around padding
        labeled_array = labeled_array[1:-1, :]
        
        # find the largest connected component
        largest_cc_size = 0
        for i in range(1, num_features + 1):
            cc_size = np.sum(labeled_array == i)
            if cc_size > largest_cc_size:
                largest_cc_size = cc_size

        receptive_field_size = largest_cc_size / ratemap.size  # percent of total map size

        return receptive_field_size
    
    def _measure_receptive_field_size(self, ratemap):
        
        normal_rf_size = self._calc_receptive_field_size(ratemap)
        inv_ratemap = self._invert_ratemap(ratemap)
        inverted_rf_size = self._calc_receptive_field_size(inv_ratemap)

        passes = inverted_rf_size < normal_rf_size

        return normal_rf_size, inverted_rf_size, passes
    
    def identify_inverse_responses(self, inv_criteria_thresh=2):

        inv_criteria_out = {}

        N_cells = self.rate_maps.shape[0]
        self.is_IEBC = np.zeros(N_cells, dtype=bool)

        for c in tqdm(range(N_cells)):
            ratemap = self.rate_maps[c,:,:]

            skew_val, skew_pass = self._measure_skewness(ratemap)

            normal_dispersion, inverted_dispersion, disp_pass = self._measure_dispursion(ratemap)

            normal_rf_size, inverted_rf_size, rf_pass = self._measure_receptive_field_size(ratemap)

            pass_count = sum([skew_pass, disp_pass, rf_pass])
            if pass_count >= inv_criteria_thresh:
                self.is_IEBC[c] = True

            inv_criteria_out['cell_{:03d}'.format(c)] = {
                'skewness': skew_val,
                'dispersion': {
                    'normal': normal_dispersion,
                    'inverted': inverted_dispersion,
                    'passes': disp_pass
                },
                'rf_size': {
                    'normal': normal_rf_size,
                    'inverted': inverted_rf_size,
                    'passes': rf_pass
                }
            }
        
        self.inv_criteria_out = inv_criteria_out

        return self.is_IEBC
    
    def _calc_mean_resultant(self, ratemap):

        N_angular_bins, N_distance_bins = ratemap.shape
        angs_rad = np.deg2rad(np.arange(0, 360, self.ray_width))

        # create a meshgrid of angles and distances
        angs_mesh, dist_mesh = np.meshgrid(angs_rad, self.dist_bin_cents, indexing='ij')

        # calculate the mean resultant vector
        mr = np.sum(ratemap * np.exp(1j * angs_mesh)) / (N_angular_bins * N_distance_bins)
        
        mean_resultant_length = np.abs(mr)
        mean_resultant_angle = np.arctan2(np.imag(mr), np.real(mr))

        if mean_resultant_angle < 0:
            mean_resultant_angle += 2*np.pi

        return mr, mean_resultant_length, mean_resultant_angle

    def _calc_single_ratemap_subsetting(self, c, inds):

        spikes = self.data['norm_spikes'][c, inds]
        ray_distances = self.ray_distances[inds, :]

        N_angular_bins = int(360 / self.ray_width)
        N_distance_bins = len(self.dist_bin_edges) - 1

        rate_map = np.zeros((N_angular_bins, N_distance_bins))

        for f in range(len(spikes)):
            for a, ang in enumerate(np.arange(0, 360, self.ray_width)):
                for d, dist_bin_start in enumerate(self.dist_bin_edges[:-1]):
                    dist_bin_end = dist_bin_start + self.dist_bin_size
                    if (ray_distances[f, a] >= dist_bin_start) and (ray_distances[f, a] < dist_bin_end):
                        rate_map[a, d] += spikes[f]

        rate_map /= self.occupancy + 1e-6

        return rate_map

    def _calc_correlation_across_split(self, c, ncnk=20, corr_thresh=0.6):

        _len = np.sum(self.useinds)

        cnk_sz = _len // ncnk

        _all_inds = np.arange(0,_len)

        chunk_order = np.arange(ncnk)
        np.random.shuffle(chunk_order)

        split1_inds = []
        split2_inds = []

        for cnk_i, cnk in enumerate(chunk_order[:(ncnk//2)]):
            _inds = _all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
            split1_inds.extend(_inds)

        for cnk_i, cnk in enumerate(chunk_order[(ncnk//2):]):
            _inds = _all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
            split2_inds.extend(_inds)

        split1_inds = np.array(np.sort(split1_inds)).astype(int)
        split2_inds = np.array(np.sort(split2_inds)).astype(int)

        rm1 = self._calc_single_ratemap_subsetting(c, split1_inds)
        rm2 = self._calc_single_ratemap_subsetting(c, split2_inds)

        # calculate 2D correlations
        corr = fm2p.corr2_coeff(rm1, rm2)
        passes = corr > corr_thresh

        return corr, passes
    
    def _test_mean_resultant_across_shuffles_mp(self, c, mrl, n_shfl=100, mrl_thresh_position=99):
 
        n_proc = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(processes=n_proc)

        spikes = self.data['norm_spikes'][c,:].copy()

        mp_param_set = [
            pool.apply_async(
                calc_shfl_mean_resultant_mp,
                args=(
                    spikes,
                    self.useinds,
                    self.occupancy,
                    self.ray_distances,
                    self.ray_width,
                    self.dist_bin_edges,
                    self.dist_bin_size,
                    self.dist_bin_cents,
                    self.is_IEBC
                )
            ) for n in range(n_shfl)
        ]
        shuffled_mrls = [result.get() for result in mp_param_set]

        shuffled_mrls = np.array(shuffled_mrls)
        use_mrl_thresh = np.percentile(shuffled_mrls, mrl_thresh_position)
        passes = mrl > use_mrl_thresh

        return use_mrl_thresh, passes

    
    def _test_mean_resultant_across_shuffles(self, c, mrl, n_shfl=100, mrl_thresh_position=99, use_mp=True):

        if use_mp:
            return self._test_mean_resultant_across_shuffles_mp(c, mrl, n_shfl, mrl_thresh_position)
            
        N_frames = np.sum(self.useinds)

        shuffled_mrls = []
        for shf in range(n_shfl):
            shift_amount = np.random.randint(int(0.1*N_frames), int(0.9*N_frames))
            shifted_spikes = np.roll(self.data['norm_spikes'][c, self.useinds], shift_amount)
            shifted_ratemap = self._calc_single_ratemap_subsetting(c, np.arange(N_frames))
            if self.is_IEBC[c]:
                shifted_ratemap = self._invert_ratemap(shifted_ratemap)
            _, shf_mrl, _ = self._calc_mean_resultant(shifted_ratemap)
            shuffled_mrls.append(shf_mrl)
        shuffled_mrls = np.array(shuffled_mrls)
        use_mrl_thresh = np.percentile(shuffled_mrls, mrl_thresh_position)
        passes = mrl > use_mrl_thresh

        return use_mrl_thresh, passes
    
    def identify_boundary_cells(self, n_chunks=20, n_shuffles=50, corr_thresh=0.6, mp=True):

        self.save_props = {}

        N_cells = self.rate_maps.shape[0]
        self.is_EBC = np.zeros(N_cells, dtype=bool)

        for c in tqdm(range(N_cells)):

            ratemap = self.rate_maps[c,:,:]

            if self.is_IEBC[c]:
                ratemap = self._invert_ratemap(ratemap)
            
            mr, mrl, mra = self._calc_mean_resultant(ratemap)

            # correlation coefficient criteria
            corr, corr_pass = self._calc_correlation_across_split(c, ncnk=n_chunks, corr_thresh=corr_thresh)

            # mean resultant criteria
            mrl_99_pctl, mrl_pass = self._test_mean_resultant_across_shuffles(c, mrl, n_shfl=n_shuffles)

            if corr_pass and mrl_pass:
                self.is_EBC[c] = True

            self.save_props['cell_{:03d}'.format(c)] = {
                'mean_resultant': mr,
                'mean_resultant_length': mrl,
                'mean_resultant_angle': mra,
                'corr_coeff': corr,
                'corr_pass': corr_pass,
                'mrl_99_pctl': mrl_99_pctl,
                'mrl_pass': mrl_pass
            }

        return self.save_props
    
    def identify_responses(self, use_angle='head', use_light=False, use_dark=False):
        
            
        if use_light:
            assert self.data['ltdk']==True, 'Data must be preprocessed with light/dark conditions.'
            print('  -> Calculating boundary responses for light condition.')
            useinds = self.data['ltdk_state_vec'].copy() == 1

        elif use_dark:
            assert self.data['ltdk'] is True, 'Data must be preprocessed with light/dark conditions.'
            print('  -> Calculating boundary responses for dark condition.')
            useinds = self.data['ltdk_state_vec'].copy() == 0

        elif (not use_light) and (not use_dark):
            print('  -> Calculating boundary responses for all frames.')
            useinds = np.ones(self.data['norm_spikes'].shape[1])

        self.useinds = useinds

        # shift spike by -2 frames
        self.data['norm_spikes'] = np.roll(self.data['norm_spikes'], -2, axis=1)

        # calculate potential angles
        if use_angle == 'head':
            self.calc_allo_yaw()

        elif use_angle == 'pupil':
            self.calc_allo_pupil()

        # calculate all ray distances
        print('  -> Calculating ray distances.')
        _ = self.get_ray_distances(angle=use_angle)

        print('  -> Calculating occupancy.')
        _ = self.calc_occupancy()

        print('  -> Calculating rate maps.')
        _ = self.calc_rate_maps()

        print('  -> Smoothing rate maps (just for later visualization).')
        _ = self.smooth_rate_maps()

        print('  -> Identifying inverse boundary cells.')
        _ = self.identify_inverse_responses()

        print('  -> Identifying boundary cells.')
        _ = self.identify_boundary_cells()

        data_out = {
            'occupancy': self.occupancy,
            'rate_maps': self.rate_maps,
            'smoothed_rate_maps': self.smoothed_rate_maps,
            'is_IEBC': self.is_IEBC,
            'is_EBC': self.is_EBC,
        }
        data_out = {**data_out, **self.save_props **self.inv_criteria_out}
        self.data_out = data_out

        return data_out

    def save_results(self, savepath):

        fm2p.write_h5(savepath, self.data_out)


if __name__ == '__main__':

    data = fm2p.read_h5(r'K:\Mini2P\250630_DMM_DMM037_ltdk\fm3\250630_DMM_DMM037_fm_03_preproc.h5')

    bt = BoundaryTuning(data)
    bt.identify_responses(use_angle='head', use_light=True)
    bt.save_results(r'K:\Mini2P\250630_DMM_DMM037_ltdk\fm3\250630_DMM_DMM037_fm_03_EBC_results_v1.h5')

# shift spikes by -2 frames

# calculate allocentric head yaw
# calculate allocentric pupil orientation

# calculate all ray distances
# shape will be (N_frames, N_rays)
# 35 cm max distance
# 2.5 cm distance bins
# 120 3 deg bins
# calculate bin center psitions

### calc occupancy
# init matrix as (N_angular_bins, N_distance bins), or (120, 14)

# for each frame
#   for each distance bin
#     which bin does the distance fall onto, as a one-hot encoded vector

### calc rate map
# init matrix as (N_cells, N_cells, N_angular_bins, N_distance_bins), or (N_cells, N_cells, 120, 14)
# for each cell
#   for each frame
#     make a mask of where distance > lower bound and distance < upper bound
#     every bin that is occupied has its value incrimented by current spike rate
# result should be (N_cells, n_frames, N_angular_bins, N_distance_bins)

# smooth the rate maps (for visualization only)
# to avoid edge effects, concatenate three copies of the rate map along the angular axis
# i.e., np.vstack((rate_map, rate_map, rate_map))
# using scipy.ndimage.gaussian_filter, smooth the rate map
# then, slice the middle third of the smoothed rate map to get the final smoothed rate map

# plot using pcolormesh

### identify inverse responses
# to identify reliable cells, first identify the inverse egocentric boundary cells
# invert all rate maps as (max - map + min)

# (1) measure skewness of rate map
# scipy.stats.skew(ratemap.flatten())
# if skew is negative (few low values nad alot of high values), this means its an IEBC

# (2) measure the dispersion of the rate map
# convert spatial bins to cartesian coordinates in egocentric space
# which bins are in the top 10% of the map?
# look through all combinations of spatial bins.
# if inverted dispersion is lower than normal dispersion, it's an IEBC

# (3) measure receptive field size
# to avoid edge effects, pad each end of the angular axis with the other end of the angular axis
# find largest continuous area of spatial bins using scipy.ndimage.label with 8-connectivity (can be diagonal)
# by initiating a 3x3 matrix
# then, drop the wrap around
# retain spatial bins above 50th percentile.
# calculate the percent of the ratemap spanned by the recpetive field
# make a copy of the rate map, invert it, and repeat above steps
# if the inverted RF size is < normal RF size, it's an IEBC

# if two of these criteria were met, call it an IEBC


### identify the EBCs
# implement equation from page 15 of andy's paper to calculate mean resultant
# how concentrated are points on a ciruclar scale?
# (Sum_{ang}^{n} \Sum_{dist}^{m} F_{ang, dist} * e^{8*ang}) / (N*M) where F is the
# firing rate in a given orientation-by-distance bin, n is the num angular bins, m is
# the number of distance bins, e is the euler constant, and i is the imaginary constant
# mean resultant length is the abs(MR)
# mean resultant angle is the arctan2(np.imag(MR), np.real(MR)), then add 2*pi to avoid negative angles

# to get preferred distance, find argmax across distance bins for each angular bin

# for c in cells:
#   if it's an IEBC, invert the rate map before continuing
#   calculate the mean resultant length and angle

#   (1) correlation coefficient criteria
#   split the recording into 20 chunks, randomly sort them into two groups, and calculate the EBC rate map for each half.
#   then, calculate the pearson correlation coefficient between the two rate maps
#   if the correlation coefficient is above 0.6, it's an EBC

#   (2) mean resultant criteria
#   for 100 iterations, randomly shift the spikes relative to behavior data by at least 10% and less than 90% of the total number of frames
#   for each shift, calculate the MRL.
#   get the 99th percentile of the MRLs.
#   see if the MRL of the original rate map is above the 99th percentile of the shuffled MRLs.
#   if it is, it's an EBC

# Repeat all of this for the light and dark conditions of the recording, seperately