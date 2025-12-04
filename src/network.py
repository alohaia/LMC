from textwrap import fill
from time import time
from datetime import datetime as dt

import pandas as pd
import numpy as np

import igraph
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Point, Polygon

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
  xy_DA_roots, max_nr_of_new_DAs=50,

  ghp_mode=2, ghp_frame_width=800, ghp_hull_meshsize=500, ghp_shape_simplifier=10,

  sample_min_DA_distance=150., newDA_prior_target_density=4.0, max_nr_of_tries=1000,
  write_refinement_steps_to_file=False, show_MCA_ACA_root=False,
  write_init_final_distribution=False,
  xy_MCA_root=np.array([]), xy_ACA_root=np.array([]),
  return_voronoi_polygon_vs_coords = False,
  filepath_folder=""
):
  """Refine the SA network, as descripted in
    S1 Appendix. Refinement of surface artery network.

  Args:

  Returns:

  """

  # Prepare array for new DA coordinates
  # xy_DA_roots_new_points = np.empty((0, 2))
  # is_initial_DA = np.full(xy_DA_roots.shape[0], fill_value=True)

  # Voronoi tessalation with DA root coordinates
  # Create ghost points as boundary
  xy_ghost_points = get_ghost_points(
    xy_DA_roots,
    ghostpoint_mode=ghp_mode,
    frame_width=ghp_frame_width,
    convex_hull_meshsize=ghp_hull_meshsize,
    shape_simplifier=ghp_shape_simplifier
  )

  # vor_initial, is_ghost_pt_initial = voronoi_tessalation(
  #   xy_DA_roots, use_ghost_points=True,
  #   xy_ghost_points=xy_ghost_points
  # )

  return [1, 2]
#
#     # vor = vor_initial
#
#     for refinementstep in range(max_nr_of_new_DAs+1):
#
#         vor, is_ghost_pt = do_voronoi_tessalation(xy_DA_roots, use_ghost_points=True,
#                                                   xy_ghost_points=xy_ghost_points)  #
#
#         _, area_vor_input_point_based = get_areas_voronoi_polygons(vor)  # area of each polygon: sorted in order of input points
#
#         xy_new_point, is_valid_point = sample_new_DA_point(vor, vor_initial, is_ghost_pt, is_ghost_pt_initial, sample_min_DA_distance,
#                                                            newDA_prior_target_density, nr_of_tries=max_nr_of_tries)
#
#         if is_valid_point and refinementstep < max_nr_of_new_DAs:
#
#             _, voronoi_region_ptid_new_pt = find_voronoi_region_for_probepoint(vor, is_ghost_pt, xy_new_point)
#
#             if write_refinement_steps_to_file or (write_init_final_distribution and refinementstep == 0):
#                 visualize_DAs_refinement(xy_DA_roots, is_initial_DA, vor, is_ghost_pt,
#                                          show_areas=True, show_MCA_ACA_root=show_MCA_ACA_root,
#                                          xy_MCA_root=xy_MCA_root, xy_ACA_root=xy_ACA_root,
#                                          area_vor_input_point_based=area_vor_input_point_based, show_new_point=True,
#                                          xy_new_point=xy_new_point,
#                                          voronoi_region_ptid_new_pt=voronoi_region_ptid_new_pt,
#                                          show_vs_ids=False, title="DAs refinement step " + str(refinementstep),
#                                          filepath=filepath_folder + "refinement_nr_" + str(refinementstep) + ".png")
#
#                 areas_valid = area_vor_input_point_based[
#                     np.logical_and(area_vor_input_point_based > 0, np.logical_not(is_ghost_pt))]
#                 visualize_DA_distribution(areas_valid,
#                                           filepath_density=filepath_folder + "density_distr_refinement_nr_" + str(refinementstep) + ".png",
#                                           filepath_area=filepath_folder + "area_distr_refinement_nr_" + str(refinementstep) + ".png")
#
#             xy_DA_roots = np.concatenate((xy_DA_roots, xy_new_point.reshape(-1, 2)), axis=0)
#             is_initial_DA = np.append(is_initial_DA, False)
#             xy_DA_roots_new_points = np.concatenate((xy_DA_roots_new_points, xy_new_point.reshape(-1, 2)), axis=0)
#
#         else:
#
#             if write_refinement_steps_to_file or write_init_final_distribution:
#                 visualize_DAs_refinement(xy_DA_roots, is_initial_DA, vor, is_ghost_pt,
#                                          show_areas=True, show_MCA_ACA_root=show_MCA_ACA_root,
#                                          xy_MCA_root=xy_MCA_root, xy_ACA_root=xy_ACA_root,
#                                          area_vor_input_point_based=area_vor_input_point_based, show_new_point=False,
#                                          xy_new_point=xy_new_point,
#                                          show_vs_ids=False, title="DAs refinement step " + str(refinementstep),
#                                          filepath=filepath_folder + "refinement_nr_" + str(refinementstep) + ".png")
#                 areas_valid = area_vor_input_point_based[
#                     np.logical_and(area_vor_input_point_based > 0, np.logical_not(is_ghost_pt))]
#                 visualize_DA_distribution(areas_valid,
#                                           filepath_density=filepath_folder + "density_distr_refinement_nr_" + str(refinementstep) + ".png",
#                                           filepath_area=filepath_folder + "area_distr_refinement_nr_" + str(refinementstep) + ".png")
#
#             print("No valid new point found at refinement step", refinementstep)
#             break
#
#     if return_voronoi_polygon_vs_coords:
#         vor_out, is_ghost_pt_out = do_voronoi_tessalation(xy_DA_roots, use_ghost_points=True,
#                                                           xy_ghost_points=xy_ghost_points)
#         polygon_vs_xy = []
#         for r in range(len(vor_out.point_region)):
#             region = vor_out.regions[vor_out.point_region[r]]
#             if not -1 in region and not is_ghost_pt_out[r]:
#                 polygon_vs_xy_current = vor_out.vertices[region]
#                 polygon_vs_xy.append(polygon_vs_xy_current)
#         return xy_DA_roots_new_points, polygon_vs_xy
#     else:
#         return xy_DA_roots_new_points

def get_ghost_points(
  xy_points: np.typing.NDArray[np.float64],
  ghostpoint_mode: int=0,
  frame_width: float=800.0,
  convex_hull_meshsize: float=200.0,
  shape_simplifier: float=10
):
  """Get ghost points as boundary for Voronoi tessalation.

  Args:
    xy_points (np.typing.NDArray[np.float64]): N*2 array of original points.

  Returns:
    ret_type: ret_desc
  """


  if ghostpoint_mode == 0:
    print("Ghost point mode 0:\n\tDo not create any ghost points.")
    xy_ghost_points = np.empty((0, 2))

  elif ghostpoint_mode == 1:
    print("Ghost point mode 1:\n"
      "\tBased on convex hull with constant frame width.\n"
      f"\tFrame width = {frame_width}\n"
      f"\tMeshsize of boundary hull = {convex_hull_meshsize}")

    hull = ConvexHull(xy_points)

    vertices_hull_ccw = hull.vertices
    nr_of_hull_vs = np.size(hull.vertices)

    xy_frame_corner_pts = np.array([])

    for i in range(nr_of_hull_vs):
      xy_vs_1 = xy_points[vertices_hull_ccw[i], :]

      if i + 1 < nr_of_hull_vs:
        xy_vs_2 = xy_points[vertices_hull_ccw[i + 1], :]
      else:
        xy_vs_2 = xy_points[vertices_hull_ccw[0], :]  # last vertex is first vertex, to close hull

        normal_vector_12 = (xy_vs_2 - xy_vs_1) / np.linalg.norm(xy_vs_2 - xy_vs_1)
        orthogonal_normal_vector_12 = np.array([normal_vector_12[1], -normal_vector_12[0]])

        xy_frame_vs_1 = xy_vs_1 + frame_width * orthogonal_normal_vector_12
        xy_frame_vs_2 = xy_vs_2 + frame_width * orthogonal_normal_vector_12

        xy_frame_corner_pts = np.append(xy_frame_corner_pts, xy_frame_vs_1)
        xy_frame_corner_pts = np.append(xy_frame_corner_pts, xy_frame_vs_2)

        xy_frame_corner_pts = xy_frame_corner_pts.reshape(-1, 2)
        nr_of_frame_corner_pts = np.size(xy_frame_corner_pts, 0)

        x_coords_subpts_frame = np.array([])
        y_coords_subpts_frame = np.array([])

        for i in range(nr_of_frame_corner_pts):

          j = i + 1  # next pt in ccw
          if j == nr_of_frame_corner_pts:
            j = 0  # to close window

            length_border_line = np.linalg.norm(xy_frame_corner_pts[i, :] - xy_frame_corner_pts[j, :])
            nr_of_subpts = np.ceil(length_border_line / convex_hull_meshsize).astype(int) + 1

            x_coords_subpts_frame = np.append(x_coords_subpts_frame,
                                              np.linspace(xy_frame_corner_pts[i, 0], xy_frame_corner_pts[j, 0],
                                                          nr_of_subpts))
            y_coords_subpts_frame = np.append(y_coords_subpts_frame,
                                              np.linspace(xy_frame_corner_pts[i, 1], xy_frame_corner_pts[j, 1],
                                                          nr_of_subpts))

        xy_ghost_points = np.array([x_coords_subpts_frame, y_coords_subpts_frame]).transpose()

  elif ghostpoint_mode == 2:
    print("Ghost point mode 2:\n"
      "\tBased on the union of circles with midpoint xy_points and radius R.\n"
      f"\tRadius = {frame_width}\n"
      f"\tResolution Simplifier = {shape_simplifier}\n")

    radius = frame_width

    nr_of_DAs = np.size(xy_points, 0)

    i = 0
    union_shapes = Point(xy_points[i, 0], xy_points[i, 1]).buffer(radius)

    i += 1
    while i < nr_of_DAs:
      union_shapes = union_shapes.union(Point(xy_points[i, 0], xy_points[i, 1]).buffer(radius))
      i += 1

      simple_pts = union_shapes.simplify(shape_simplifier, preserve_topology=True)
      pts_np = np.array(simple_pts.exterior.coords.xy).transpose()

      xy_ghost_points = np.array([pts_np[:, 0], pts_np[:, 1]]).transpose()

    else:
      print("Ghost point mode default: Do not create any ghost points.")
      xy_ghost_points = np.empty((0, 2))

    return xy_ghost_points


def voronoi_tessalation(xy_DA_roots, use_ghost_points=False, xy_ghost_points=np.array([[]])):
    if use_ghost_points:
        xy_voronoi_pts = np.concatenate((xy_DA_roots, xy_ghost_points), axis=0)
    else:
        xy_voronoi_pts = xy_DA_roots

    is_ghost_pt = np.array([False] * np.size(xy_voronoi_pts, 0))
    is_ghost_pt[np.size(xy_DA_roots, 0):] = True

    vor = Voronoi(xy_voronoi_pts)

    return vor, is_ghost_pt


# def get_areas_voronoi_polygons(vor):
#     coords_voronoi_vertices = vor.vertices  # coordinates of vertices forming the polygon
#     area_vor_regions = np.array([])  # surface area of each polygon
#
#     # find area of each voronoi polygon
#     for i in vor.regions:  # i is list of vertices forming each region
#         if len(i) > 2 and (-1 not in i):  # if -1: not closed, if only 2 vertices: also not closed
#             x_poly = coords_voronoi_vertices[i, 0]
#             y_poly = coords_voronoi_vertices[i, 1]
#             area_vor_regions = np.append(area_vor_regions, PolyArea(x_poly, y_poly))
#         else:
#             area_vor_regions = np.append(area_vor_regions, -1)
#
#     nr_of_points = np.size(vor.point_region)
#     area_vor_input_point_based = np.zeros(nr_of_points)
#     for i in xrange(nr_of_points):
#         current_region = vor.point_region[i]
#         area_vor_input_point_based[i] = area_vor_regions[current_region]
#     # new criteria
#     return area_vor_regions, area_vor_input_point_based  # areas for each polygon: for each region, for each associated point
#
#
# def is_in_voronoi_region(vor, region, probe_point):
#     # check if a certain probe point is within a certain vor region
#
#     if np.size(probe_point) != 2:
#         print "Probe point not valid"
#
#     current_region_vs = vor.regions[region]
#     xy_current_region_vs = vor.vertices[current_region_vs]
#
#     poly = Polygon(xy_current_region_vs)
#     probe_point = Point(probe_point)
#
#     return probe_point.within(poly)
#
#
# def sample_new_DA_point(vor, vor_initial, is_ghost_point, is_ghost_pt_initial, min_distance, target_density, nr_of_tries=10):
#     x_min = np.min(vor_initial.points[:, 0])
#     x_max = np.max(vor_initial.points[:, 0])
#     y_min = np.min(vor_initial.points[:, 1])
#     y_max = np.max(vor_initial.points[:, 1])
#
#     is_valid_point = False
#
#     xy_new_pt = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), ])
#
#     it = 0
#
#     xy_current_points_DAs = vor.points[np.logical_not(is_ghost_point)]
#
#     while (not is_valid_point) and it < nr_of_tries:
#
#         it += 1
#
#         xy_new_pt = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), ])
#         _, initial_vor_region_pt_id = find_voronoi_region_for_probepoint(vor_initial, is_ghost_pt_initial, xy_new_pt)
#         if initial_vor_region_pt_id < 0:  # not in range which is allowed from initial geometry
#             continue  # sample new point
#         else:
#             _, vor_region_pt_id = find_voronoi_region_for_probepoint(vor, is_ghost_point, xy_new_pt)
#             _, areas_vor_point_based = get_areas_voronoi_polygons(vor)
#
#             current_density = 1.e6 / areas_vor_point_based[vor_region_pt_id]
#             closest_distance = get_closest_distance(xy_current_points_DAs, xy_new_pt)
#
#             if current_density < target_density and closest_distance > min_distance:  # and distance to DAs
#                 is_valid_point = True
#             else:
#                 continue
#
#     if is_valid_point:
#         print "New DA coords: ", xy_new_pt, "Nr. of tries:", it, "Is valid DA:", is_valid_point
#         return xy_new_pt, is_valid_point
#     else:
#         print "No valid DA found after", it, "tries..."
#         return np.array([-1.e9, -1.e9]), is_valid_point
#
#
# def find_voronoi_region_for_probepoint(vor, is_ghost_point, probe_point):
#     # returns the region and the corresponding point id of a probe point
#
#     regions = vor.point_region
#     nr_of_points = np.size(vor.point_region)
#     for point_id in np.arange(nr_of_points):
#         current_region_id = regions[point_id]
#         if is_ghost_point[point_id] or len(vor.regions[current_region_id]) < 3 or (
#                 -1 in vor.regions[current_region_id]):
#             continue
#         else:
#             is_in_region = is_in_voronoi_region(vor, current_region_id, probe_point)
#             if is_in_region:
#                 return current_region_id, point_id
#             else:
#                 continue
#
#     return -1, -1
