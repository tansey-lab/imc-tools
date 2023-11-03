import geopandas
import tqdm
from scipy.ndimage import center_of_mass
import numpy as np
from shapely import Polygon, Point
from geopandas import GeoDataFrame
from scipy.spatial import Voronoi

from imc_tools.images import raster_to_polygon, contour_triangle_to_poly
from imc_tools.mcd import extract_channel_to_arr, get_used_channels_for_acquisition
import cv2


def mask_to_centroids(segmented_image):
    idx = np.unique(segmented_image)
    idx = idx[idx != 0]

    gdf = GeoDataFrame(index=idx, columns=['nucleus_centroid', 'nucleus'])

    unique_segments = np.unique(segmented_image)

    for unique_segment in unique_segments:
        if unique_segment == 0:
            continue
        raster = (segmented_image == unique_segment).astype(np.uint8)
        center = Point(*center_of_mass(raster))
        nucleus = raster_to_polygon(raster)

        gdf['nucleus_centroid'].loc[unique_segment] = center
        gdf['nucleus'].loc[unique_segment] = nucleus

    return gdf


def get_voronoi_cell_shapes(vor, bounding_polygon):
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if not -1 in region:  # Bounded regions
            polygon = Polygon([vor.vertices[i] for i in region])
            clipped_polygon = polygon.intersection(bounding_polygon)
            yield clipped_polygon
        else:  # Unbounded regions
            polygon_vertices = [vor.vertices[i] for i in region if i != -1]
            if len(polygon_vertices) < 3:  # Skip invalid polygons
                continue
            polygon = Polygon(polygon_vertices)
            clipped_polygon = polygon.intersection(bounding_polygon)
            if clipped_polygon.is_empty:  # No intersection
                continue
            yield clipped_polygon


def add_voronoi_cell_bodies(gdf, bounding_polygon):
    points_as_tuples = np.array([(a.x, a.y) for a in gdf.nucleus_centroid])
    gdf['voronoi_cell_body'] = None

    for poly in get_voronoi_cell_shapes(Voronoi(points_as_tuples), bounding_polygon):
        selector = gdf.set_geometry("nucleus_centroid").within(poly)

        if selector.sum() == 0:
            continue
        elif selector.sum() == 1:
            gdf['voronoi_cell_body'].loc[selector] = poly
        else:
            raise RuntimeError("Multiple nuclei in one voronoi cell")

    gdf['voronoi_cell_area'] = gdf.set_geometry('voronoi_cell_body').area


def get_gdf_for_constrained_triangulation(result):
    polys = [contour_triangle_to_poly(x) for x in result.triangles()]
    gdf = geopandas.GeoDataFrame(index=range(len(polys)), geometry=polys)
    return gdf


def nucleus_mask_to_voronoi_cell_bodies(segmented_image):
    bounding_polygon = Polygon([(0, 0),
                                (0, segmented_image.shape[1] - 1),
                                (segmented_image.shape[0] - 1, segmented_image.shape[1] - 1),
                                (segmented_image.shape[0] - 1, 0)])

    gdf = mask_to_centroids(segmented_image)
    add_voronoi_cell_bodies(gdf, bounding_polygon)
    return gdf


def polygon_to_mask(poly, area):
    mask = np.zeros_like(area).astype(np.uint8)

    # get list of border points as tuples from poly
    border_points = np.array(poly.exterior.coords).astype(np.int32)

    # Fill the polygon in the mask
    cv2.fillPoly(mask, [border_points], 1)

    return mask.astype(bool)


def add_total_signal_to_cell_body_gdf(mcd_fn, gdf):
    channels = get_used_channels_for_acquisition(mcd_fn)

    for channel in tqdm.tqdm(channels):
        signal_arr = extract_channel_to_arr(mcd_fn, channel, slide_idx=0, acquisition_idx=0)

        def get_pseudocount(poly):
            mask = polygon_to_mask(poly, signal_arr)
            return np.sum(signal_arr[mask])

        gdf[f"{channel}__pseudocount"] = gdf['voronoi_cell_body'].apply(lambda x: get_pseudocount(x) if x else None)
