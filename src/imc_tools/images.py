import logging
import os

from PIL import Image
from cv2.ximgproc import createStructuredEdgeDetection

from scipy.stats import norm
from scipy.optimize import minimize_scalar
import numpy as np
import cv2

from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

PRETRAINED_SED_MODEL_PATH = os.path.join(SRC_DIR, "model.yml.gz")

def remove_outliers(arr, percentile=99.5):
    arr = arr.copy()
    max_value = np.percentile(arr, percentile)
    arr[arr > max_value] = max_value
    return arr


def normalize(arr):
    return arr / np.max(arr)


def greyscale_to_colored_transparency(
    greyscale_image, color=(57, 255, 20), fixed_alpha=None
):
    arr = np.array(greyscale_image)
    alpha_channel = arr  # Alpha from grayscale image
    rgba_overlay = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba_overlay[:, :, 0] = color[0]
    rgba_overlay[:, :, 1] = color[1]
    rgba_overlay[:, :, 2] = color[2]
    if fixed_alpha is not None:
        rgba_overlay[:, :, 3] = fixed_alpha
    else:
        rgba_overlay[:, :, 3] = alpha_channel
    return Image.fromarray(rgba_overlay, "RGBA")


def black_to_alpha(image):
    image_array = np.array(image)

    # Calculate "blackness" based on RGB values (average for this example)
    new_alpha = np.max(image_array[:, :, :3], axis=2).astype(np.uint8)

    # Assign the new alpha channel back to the image array
    image_array[:, :, 3] = new_alpha

    # Convert the NumPy array back to a PIL image and save
    return Image.fromarray(image_array, "RGBA")


def overlay_arrays_as_images(arr1, arr2):
    img1 = Image.fromarray(arr1).convert("L")
    img2 = Image.fromarray(arr2).convert("L")

    img1_color = greyscale_to_colored_transparency(img1, color=(255, 0, 0))
    img2_color = greyscale_to_colored_transparency(img2, color=(0, 255, 0))

    overlay = Image.alpha_composite(img1_color, img2_color)
    return overlay


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def get_image_gradient(img, model_path=PRETRAINED_SED_MODEL_PATH):
    model = createStructuredEdgeDetection(model_path)

    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    gradient = model.detectEdges(img.astype(np.float32) / 255.0)
    return (gradient / gradient.max() * 255).astype(np.uint8)

def get_segmented_image(gradient_image, num_regions = 100):
    import higra as hg
    graph = hg.get_4_adjacency_graph(gradient_image.shape[:2])
    edge_weights = hg.weight_graph(graph, gradient_image, hg.WeightFunction.mean)
    tree, altitudes = hg.watershed_hierarchy_by_area(graph, edge_weights)
    explorer = hg.HorizontalCutExplorer(tree, altitudes)
    mode = "At least n regions"
    cut_nodes = explorer.horizontal_cut_from_num_regions(num_regions, at_least=(mode == 'At least n regions'))
    random_color = np.random.rand(tree.num_vertices(), 3)
    im = cut_nodes.reconstruct_leaf_data(tree, random_color)
    flattened_im = im.reshape((im.shape[0]*im.shape[1],im.shape[2]))
    uniq_im = np.unique(flattened_im,axis =0)
    mask = np.zeros(flattened_im.shape[0])

    count = 1
    for im_val in uniq_im:
        ind = np.all(flattened_im==im_val,axis=1)
        mask[ind]=count
        count+=1

    mask = mask.reshape(gradient_image.shape)
    return mask.astype(np.uint32)


def get_mixture_of_two_gaussians_cutoff_point(arr):
    from sklearn.mixture import GaussianMixture
    # Fit a Gaussian Mixture Model with 2 components
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(arr.reshape(-1, 1))

    # Get the parameters of the two Gaussian distributions
    # Get the parameters of the two Gaussian distributions
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    # Sort the means so we know which is the first and second Gaussian
    idx = np.argsort(means)
    means = means[idx]
    covariances = covariances[idx]
    weights = weights[idx]

    # Define the function to minimize
    def objective(x):
        p1 = weights[0] * norm.pdf(x, means[0], np.sqrt(covariances[0]))
        p2 = weights[1] * norm.pdf(x, means[1], np.sqrt(covariances[1]))
        return np.abs(p1 - p2)

    # Find the cutoff point
    result = minimize_scalar(objective, bounds=(means[0], means[1]), method='bounded')
    cutoff = result.x

    return cutoff


def raster_to_polygon(raster):
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(raster.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to hold outer contours and holes
    outer_contours = []
    holes = []

    for i, contour in enumerate(contours):
        # The contour points are in (row, column) format, so we flip them to (x, y)
        coords = contour.squeeze(axis=1)
        xy = [(p[1], p[0]) for p in coords]

        # Check if this contour has a parent
        if hierarchy[0][i][3] == -1:
            # No parent, so it's an external contour
            outer_contours.append(xy)
        else:
            # Has a parent, so it's a hole
            holes.append(xy)

    # Assuming single polygon, we take the first outer contour
    # You can loop over outer_contours to create multiple polygons if needed
    if outer_contours:
        exterior = outer_contours[0]
        interior = holes
        return Polygon(exterior, holes=interior)


import ground.core.geometries

def raster_to_sect_polygon(raster):
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(raster.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to hold outer contours and holes
    outer_contours = []
    holes = []

    for i, contour in enumerate(contours):
        # The contour points are in (row, column) format, so we flip them to (x, y)
        coords = contour.squeeze(axis=1)
        xy = [ground.core.geometries.Point(p[1], p[0]) for p in coords]

        # Check if this contour has a parent
        if hierarchy[0][i][3] == -1:
            # No parent, so it's an external contour
            outer_contours.append(ground.core.geometries.Contour(xy))
        else:
            # Has a parent, so it's a hole
            holes.append(ground.core.geometries.Contour(xy))

    # Assuming single polygon, we take the first outer contour
    # You can loop over outer_contours to create multiple polygons if needed
    if outer_contours:
        exterior = outer_contours[0]
        interior = holes
        return ground.core.geometries.Polygon(border=exterior, holes=interior)

import shapely.geometry

def contour_triangle_to_poly(contour):
    return shapely.geometry.Polygon([(p.x, p.y) for p in contour.vertices])


