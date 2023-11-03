import cv2
import numpy as np
from matplotlib import pyplot as plt, patches as mpatches

import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def plot_polygon(polygon):
    """
    Plots a single Shapely polygon.

    :param polygon: A Shapely Polygon object.
    """
    # Extract x and y coordinates from the polygon's exterior
    x, y = polygon.exterior.xy

    # Plot the polygon
    plt.figure()
    plt.fill(x, y, alpha=0.6, fc='blue', label='Polygon')
    plt.title('Polygon Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_cell_body_voronoi(gdf):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot polygons
    gdf.set_geometry('voronoi_cell_body').boundary.plot(ax=ax,
                                           color='blue',
                                           label='Voronoi Cell Body')

    # Plot points
    gdf.set_geometry('nucleus_centroid').plot(ax=ax, color='red', markersize=5, label='Nucleus Centroid')

    # Set labels, title, and legend
    plt.xlabel('X')
    plt.ylabel('Y')

    # Show the plot
    plt.show()


def plot_cell_body_pseudocount(gdf, channel_name):
    ax = gdf.set_geometry('voronoi_cell_body').plot(column=f'{channel_name}__pseudocount',
                                                    cmap='viridis',
                                                    legend=True,
                                                    figsize=(10, 10))

    # Set labels, title, and axes equal
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # flip x and y axis


    # Show the plot
    plt.show()

def plot_cell_body_area_averaged_pseudocount(gdf, channel_name):
    gdf[f'{channel_name}__avg'] = gdf[f'{channel_name}__pseudocount'] / gdf['voronoi_cell_area']

    ax = gdf.set_geometry('voronoi_cell_body').plot(column=f'{channel_name}__avg',
                                                    cmap='viridis',
                                                    legend=True,
                                                    figsize=(10, 10))

    # Set labels, title, and axes equal
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # flip x and y axis


    # Show the plot
    plt.show()



def plot_segmentation_borders(segmentation_mask, output_path):
    # Plotting
    fig, ax = plt.subplots()

    for i in np.unique(segmentation_mask):
        segment_of_interest = (segmentation_mask == i).astype(bool).astype(np.uint8)
        segment_of_interest[segment_of_interest > 0] = 255
        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(segment_of_interest.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)



        # Loop through contours and plot each one
        for contour in contours:
            # The contour points are in (row, column) format, so we flip them to (x, y)
            contour = contour.squeeze(axis=1)
            x = contour[:, 1]
            y = contour[:, 0]

            # Create a polygon and add it to the plot
            polygon = mpatches.Polygon(xy=list(zip(x, y)),
                                       fill=False,
                                       linewidth=1,
                                       color='r')
            ax.add_patch(polygon)

    # Set the limits to match the size of the raster
    ax.set_xlim(0, segmentation_mask.shape[1])
    ax.set_ylim(0, segmentation_mask.shape[0])
    ax.set_aspect('equal', 'box')

    # Invert the y-axis to match the raster's orientation
    ax.invert_yaxis()

    fig.savefig(output_path)
