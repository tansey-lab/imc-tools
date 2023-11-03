from scipy.ndimage import measurements
import numpy as np

def mask_to_centroids(segmented_image):
    unique_segments = np.unique(segmented_image)

    result = {}
    for unique_segment in unique_segments:

        raster = segmented_image[(segmented_image == unique_segment)].astype(int)
        result[unique_segment] = measurements.center_of_mass(raster)

    return result


def get_voronoi_cell_shapes():
    pass