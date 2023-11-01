import cv2
import numpy as np
import higra as hg
import os

from cv2.ximgproc import createStructuredEdgeDetection

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

PRETRAINED_SED_MODEL_PATH = os.path.join(SRC_DIR, "model.yml.gz")


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def get_image_gradient(img, model_path=PRETRAINED_SED_MODEL_PATH):
    model = createStructuredEdgeDetection(model_path)

    if img.shape == 2:
        img = np.stack([img] * 3, axis=-1)

    gradient = model.detectEdges(img.astype(np.float32) / 255.0)
    return (gradient / gradient.max() * 255).astype(np.uint8)

def get_segmented_image(gradient_image, num_regions = 100):
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