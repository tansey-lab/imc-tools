import geopandas
from scipy.ndimage import center_of_mass
import numpy as np
from shapely import Polygon, Point
from geopandas import GeoDataFrame
from scipy.spatial import Voronoi

from imc_tools.images import raster_to_polygon, contour_triangle_to_poly


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
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            voronoi_polygon = Polygon([vor.vertices[i] for i in region])

            clipped_polygon = voronoi_polygon.intersection(bounding_polygon)
            if not clipped_polygon.exterior.is_closed:
                raise RuntimeError("Voronoi cell is not closed")

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


def nucleus_mask_to_voronoi_cell_bodies(segmented_image):
    bounding_polygon = Polygon([(0, 0),
                                (0, segmented_image.shape[1]-1),
                                (segmented_image.shape[0]-1, segmented_image.shape[1]-1),
                                (segmented_image.shape[0]-1, 0)])

    gdf = mask_to_centroids(segmented_image)
    add_voronoi_cell_bodies(gdf, bounding_polygon)
    return gdf


def get_gdf_for_constrained_triangulation(result):
    polys = [contour_triangle_to_poly(x) for x in result.triangles()]
    gdf = geopandas.GeoDataFrame(index=range(len(polys)), geometry=polys)
    return gdf
