from textwrap import fill
from time import time
from datetime import datetime as dt

from matplotlib.pyplot import flag
from numpy.typing import NDArray
import pandas as pd
import numpy as np

import igraph
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.arrays import boolean
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from typing import Tuple, TypeAlias, List, Union

CoordinateArray: TypeAlias = np.typing.NDArray[np.float64]

def create(vertices: str, edges: str) -> igraph.Graph:
    """Create an igraph Graph object from vertices and edges CSV data.

    Args:
        vertices: Path to the CSV file containing the vertex data (nodes).
        edges: Path to the CSV file containing the edge data (links).

    Returns:
        igraph.Graph: The constructed igraph Graph object.
    """

    vs_data = pd.read_csv(vertices)
    es_data = pd.read_csv(edges)

    g = igraph.Graph((es_data[["from", "to"]] - 1).to_numpy().tolist())

    g['network_gen_date'] = dt.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    g['Mouse'] = "Test 1"

    vs_coords = vs_data.loc[:, ["X", "Y"]].to_numpy()
    vs_coords = np.hstack([vs_coords, np.zeros((vs_coords.shape[0], 1))])

    # vertices
    g.vs["coords"] = vs_coords
    g.vs["MCA_in"] = vs_data.MCA_in
    g.vs["ACA_in"] = vs_data.ACA_in
    g.vs["is_DA_startingPt"] = vs_data.is_DA_startingPt # FIXME
    g.vs["is_DA_startingPt_added_manually"] = np.zeros(g.vcount(), dtype=int)
    g.vs['is_AV_root'] = np.zeros(g.vcount(), dtype=int)
    g.vs['is_connected_2caps'] = np.zeros(g.vcount(), dtype=int)
    g.vs['COW_in'] = np.zeros(g.vcount(),dtype=int)

    # Edges
    g.es['is_stroke'] = es_data.is_stroke
    g.es["diameter"] = es_data.diameter
    g.es["is_collateral"] = es_data.is_collateral
    g.es['type'] = es_data.type
    g.es["added_manually"] = es_data.added_manually
    # FIXME
    g.es["index_exp"] = es_data.index_exp
    g.es["diam_pre_exp"] = es_data.diam_pre_exp
    g.es["diam_post_exp"] = es_data.diam_post_exp
    g.es["vRBC_pre_exp"] = es_data.vRBC_pre_exp
    g.es["vRBC_post_exp"] = es_data.vRBC_post_exp
    g.es["vRBC_pre_larger10"] = es_data.vRBC_pre_larger10

    lengths = []
    for e in g.es:
        v1, v2 = e.tuple
    p1, p2 = g.vs[v1]["coords"], g.vs[v2]["coords"]
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    lengths.append(dist)
    g.es["length"] = lengths

    return g


def check_attrs(graph) -> bool:
    """Check consistence of Graph attributes.

    Args:
        graph (igraph.Graph): The Graph object to be checked.

    Returns:
        bool: If the attributes are the same as required.
    """

    current_attr = {
        "graph": graph.attributes(),
        "vertex": graph.vs.attributes(),
        "edge": graph.es.attributes(),
    }

    target_attr = {
        "graph": ['Mouse', 'network_gen_date'],
        "vertex": ['ACA_in', 'MCA_in', 'COW_in', 'coords', 'is_AV_root',
                'is_DA_startingPt','is_DA_startingPt_added_manually',
                'is_connected_2caps'],
        "edge": ['added_manually', 'diam_post_exp', 'diam_pre_exp', 'diameter',
                'index_exp', 'is_collateral', 'is_stroke', 'length', 'type',
                'vRBC_post_exp', 'vRBC_pre_exp', 'vRBC_pre_larger10']
    }

    attributes_missing = []
    attributes_excessive = []
    is_consistent = True

    if (len(set(target_attr["graph"])) != len(target_attr["graph"]) or
        len(set(target_attr["vertex"])) != len(target_attr["vertex"]) or
        len(set(target_attr["edge"])) != len(target_attr["edge"])):
        is_consistent = False
        print("List of target edge attributes not unique.")
        return is_consistent

    # Check if all target attributes are in current graph
    for i in target_attr.keys():
        for attr in target_attr[i]:
            if attr in current_attr[i]:
                continue
            else:
                attributes_missing.append(attr)
                is_consistent = False

    # Check if no additional attributes are in current graph
    for i in current_attr.keys():
        for attr in current_attr[i]:
            if attr in target_attr[i]:
                continue
            else:
                attributes_excessive.append(attr)
                is_consistent = False

    if not is_consistent:
        print("Something is wrong with graph attributes:")
        print("\tMissing attribute(s) detected:", attributes_missing)
        print("\tExcessive attribute(s) detected:", attributes_excessive)

    return is_consistent

def refine_DA_density(
    xy_DA_roots: CoordinateArray,
    ghp_boundary_offset: float=800,
    ghp_simplify_tolerance: float=10,

    nr_new_DAs_max: int=50,
    sample_min_DA_distance=150.,
    newDA_prior_target_density=4.0,
    max_nr_of_tries=1000,
    save_refinement_steps=False,
    save_init_final_distribution=False,
    save_path="",
    show_MCA_ACA_root=False,
    xy_MCA_root=np.array([]), xy_ACA_root=np.array([]),
    return_voronoi_polygon_vs_coords = False
):
    """Refine the SA network, as descripted in
    S1 Appendix. Refinement of surface artery network.

    Args:

        xy_DA_roots: CoordinateArray
            Coordinates of original DA roots.

        ghp_boundary_offset: float;
            See `gen_ghost_points()`.

        ghp_simplify_tolerance: float:
            See `gen_ghost_points()`.

    Returns:

    """

    # Prepare array for new DA coordinates
    xy_DA_roots_new_points = np.empty((0, 2))
    is_initial_DA = np.full(xy_DA_roots.shape[0], fill_value=True)

    # Voronoi tessalation with DA root coordinates
    # Create ghost points as boundary
    xy_ghost_points = gen_ghost_points(
        xy_DA_roots,
        radius=ghp_boundary_offset,
        simplify_tolerance=ghp_simplify_tolerance
    )
    vor_initial, is_ghost_pt_initial = voronoi_tessalation(
        xy_DA_roots,
        xy_ghost_points=xy_ghost_points
    )

    vor, is_ghost_pt = vor_initial, is_ghost_pt_initial
    for step_refine in range(nr_new_DAs_max + 1):

        # vor, is_ghost_pt = voronoi_tessalation(
        #   xy_DA_roots,
        #   xy_ghost_points=xy_ghost_points
        # )

        # Area of each polygon, sorted in order of input points
        _, area_vor_pt_regions = get_areas_voronoi_polygons(vor)

        xy_new_point, is_valid_point = sample_new_DA_point(
            vor, vor_initial, is_ghost_pt, is_ghost_pt_initial,
            sample_min_DA_distance,
            newDA_prior_target_density,
            nr_of_tries=max_nr_of_tries
        )

        if is_valid_point and step_refine < nr_new_DAs_max:
            _, voronoi_region_ptid_new_pt = find_voronoi_region_for_probepoint(
                vor, is_ghost_pt, xy_new_point
            )

        if save_refinement_steps or (save_init_final_distribution and step_refine == 0):
            visualize_DAs_refinement(
                xy_DA_roots, is_initial_DA, vor, is_ghost_pt,
                show_areas=True,
                show_MCA_ACA_root=show_MCA_ACA_root,
                xy_MCA_root=xy_MCA_root, xy_ACA_root=xy_ACA_root,
                area_vor_input_point_based=area_vor_pt_regions,
                show_new_point=True,
                xy_new_point=xy_new_point,
                voronoi_region_ptid_new_pt=voronoi_region_ptid_new_pt,
                show_vs_ids=False,
                title="DAs refinement step " + str(step_refine),
                filepath=save_path + "refinement_nr_" + str(step_refine) + ".png"
            )

        areas_valid = area_vor_pt_regions[
        np.logical_and(area_vor_pt_regions > 0, np.logical_not(is_ghost_pt))]
        visualize_DA_distribution(
            areas_valid,
            filepath_density=f"${save_path}density_distr_refinement_nr_${step_refine}.png"
                filepath_area=f"${save_path}area_distr_refinement_nr_${step_refine}.png"
        )

        xy_DA_roots = np.concatenate((xy_DA_roots, xy_new_point.reshape(-1, 2)), axis=0)
        is_initial_DA = np.append(is_initial_DA, False)
        xy_DA_roots_new_points = np.concatenate((xy_DA_roots_new_points, xy_new_point.reshape(-1, 2)), axis=0)

    else:

        if save_refinement_steps or save_init_final_distribution:
            visualize_DAs_refinement(
                xy_DA_roots, is_initial_DA, vor, is_ghost_pt,
                show_areas=True, show_MCA_ACA_root=show_MCA_ACA_root,
                xy_MCA_root=xy_MCA_root, xy_ACA_root=xy_ACA_root,
                area_vor_input_point_based=area_vor_pt_regions, show_new_point=False,
                xy_new_point=xy_new_point,
                show_vs_ids=False, title="DAs refinement step " + str(step_refine),
                filepath=save_path + "refinement_nr_" + str(step_refine) + ".png"
            )
            areas_valid = area_vor_pt_regions[
                np.logical_and(area_vor_pt_regions > 0, np.logical_not(is_ghost_pt))
            ]
            visualize_DA_distribution(
                areas_valid,
                filepath_density=save_path + "density_distr_refinement_nr_" + str(step_refine) + ".png",
                filepath_area=save_path + "area_distr_refinement_nr_" + str(step_refine) + ".png"
            )

            print("No valid new point found at refinement step", step_refine)
            break

    if return_voronoi_polygon_vs_coords:
        vor_out, is_ghost_pt_out = voronoi_tessalation(
            xy_DA_roots, xy_ghost_points=xy_ghost_points
        )
    polygon_vs_xy = []
    for r in range(len(vor_out.point_region)):
        region = vor_out.regions[vor_out.point_region[r]]
        if not -1 in region and not is_ghost_pt_out[r]:
            polygon_vs_xy_current = vor_out.vertices[region]
        polygon_vs_xy.append(polygon_vs_xy_current)
        return xy_DA_roots_new_points, polygon_vs_xy
    else:
        return xy_DA_roots_new_points

def gen_ghost_points(
  xy_points: CoordinateArray,
  radius: float=800,
  simplify_tolerance: float=10
) -> CoordinateArray:
  """Generate ghost points as boundary for Voronoi tessalation.

  Args:
    xy_points: CoordinateArray
      N*2 array of original points.

    radius: float
      Radius to buffer points.

    simplify_tolerance: float
      Tolerance of `Polygon.simplify ` to reduce result points.

  Returns:
    CoordinateArray:
      Boundary points as ghost points.
  """
  assert xy_points.shape[1] == 2

  print("Based on the union of circles with midpoint xy_points and radius R.\n"
    f"\tRadius = {radius}\n"
    f"\tResolution Simplifying tolerance = {simplify_tolerance}\n")

  # Generate circles
  circ_list : List[Polygon] = [
    Point(x, y).buffer(radius)
    for x, y in xy_points
  ]
  # Union all circles
  union_polygon = unary_union(circ_list)
  # Simplify union_polygon to reduce boundary points
  union_polygon = union_polygon.simplify(simplify_tolerance, preserve_topology=True)

  return np.array(union_polygon.boundary.xy).T


def voronoi_tessalation(
  xy_DA_roots: CoordinateArray,
  xy_ghost_points: CoordinateArray
) -> Tuple[Voronoi, NDArray[np.bool]]:
  xy_voronoi_pts = np.concatenate((xy_DA_roots, xy_ghost_points), axis=0)

  is_ghost_pt = np.full(xy_voronoi_pts.shape[0], fill_value=False)
  is_ghost_pt[xy_DA_roots.shape[0]:] = True

  vor = Voronoi(xy_voronoi_pts)

  return vor, is_ghost_pt

# Shoelace formula for area of polygon (https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates)
def PolyArea(
    coords: CoordinateArray
) -> np.float64:
    """Description

    Args:
        arg1_name: arg1_type
            arg1_desc

    Returns:
        ret_type:
            ret_desc
    """
    assert coords.shape[1] == 2
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_areas_voronoi_polygons(
    vor: Voronoi
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get areas of each polygons in a Voronoi object.

    Args:
        vor: Voronoi
            A Voronoi object containing multiple polygons.

    Returns:
        area_vor_regions: List[np.float64]
            Areas for each region, -1 represents open geometry.

        area_vor_pt_regions: List[np.float64]
            Areas for each point of vor.vertices in the same order.
    """

    # coordinates of all vertices
    coords_voronoi_vertices : CoordinateArray = vor.vertices
    # surface area of each polygon
    area_vor_regions : NDArray[np.float64] = np.empty((0, ))

    # find area of each voronoi polygon
    # i (list[int]) is a list of indices to vertices of each region
    for i in vor.regions:
        if len(i) > 2 and (-1 not in i):  # -1 or only 2 vertices: not closed
            corrds_rg = coords_voronoi_vertices[i, :]
            area_vor_regions = np.append(area_vor_regions, PolyArea(corrds_rg))
        else:
            area_vor_regions = np.append(area_vor_regions, -1)

    nr_points = np.size(vor.point_region)
    area_vor_pt_regions : NDArray[np.float64] = np.zeros(nr_points)
    for i in range(nr_points):
        # vor.point_region[i] is the region corresponds to the vor.vertices[i]
        pt_region : np.intp = vor.point_region[i]
        area_vor_pt_regions[i] = area_vor_regions[pt_region]

    # areas for each polygon: for each region, for each associated point
    return area_vor_regions, area_vor_pt_regions


def is_in_voronoi_region(vor, region, probe_point):
    # check if a certain probe point is within a certain vor region

    if np.size(probe_point) != 2:
        print "Probe point not valid"

    current_region_vs = vor.regions[region]
    xy_current_region_vs = vor.vertices[current_region_vs]

    poly = Polygon(xy_current_region_vs)
    probe_point = Point(probe_point)

    return probe_point.within(poly)


def sample_new_DA_point(
    vor : Voronoi,
    vor_initial : Voronoi,
    is_ghost_point,
    is_ghost_pt_initial,
    min_distance,
    target_density,
    nr_of_tries=10
):
  x_min = np.min(vor_initial.points[:, 0])
  x_max = np.max(vor_initial.points[:, 0])
  y_min = np.min(vor_initial.points[:, 1])
  y_max = np.max(vor_initial.points[:, 1])

  is_valid_point = False

  xy_new_pt = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), ])

  it = 0

  xy_current_points_DAs = vor.points[np.logical_not(is_ghost_point)]

  while (not is_valid_point) and it < nr_of_tries:

    it += 1

    xy_new_pt = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), ])
    _, initial_vor_region_pt_id = find_voronoi_region_for_probepoint(vor_initial, is_ghost_pt_initial, xy_new_pt)
    if initial_vor_region_pt_id < 0:  # not in range which is allowed from initial geometry
      continue  # sample new point
      else:
      _, vor_region_pt_id = find_voronoi_region_for_probepoint(vor, is_ghost_point, xy_new_pt)
      _, areas_vor_point_based = get_areas_voronoi_polygons(vor)

      current_density = 1.e6 / areas_vor_point_based[vor_region_pt_id]
      closest_distance = get_closest_distance(xy_current_points_DAs, xy_new_pt)

      if current_density < target_density and closest_distance > min_distance:  # and distance to DAs
        is_valid_point = True
        else:
        continue

    if is_valid_point:
      print "New DA coords: ", xy_new_pt, "Nr. of tries:", it, "Is valid DA:", is_valid_point
      return xy_new_pt, is_valid_point
    else:
      print "No valid DA found after", it, "tries..."
      return np.array([-1.e9, -1.e9]), is_valid_point



def find_voronoi_region_for_probepoint(vor, is_ghost_point, probe_point):
  # returns the region and the corresponding point id of a probe point

  regions = vor.point_region
  nr_of_points = np.size(vor.point_region)
  for point_id in np.arange(nr_of_points):
    current_region_id = regions[point_id]
    if is_ghost_point[point_id] or len(vor.regions[current_region_id]) < 3 or (
      -1 in vor.regions[current_region_id]):
      continue
      else:
      is_in_region = is_in_voronoi_region(vor, current_region_id, probe_point)
      if is_in_region:
        return current_region_id, point_id
        else:
        continue

    return -1, -1
