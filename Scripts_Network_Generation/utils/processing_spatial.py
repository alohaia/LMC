import numpy as np
from scipy.spatial import ConvexHull, Voronoi
from shapely.geometry import Point, Polygon

from .network_visz import (
    visualize_DA_distribution,
    visualize_DAs_refinement,
    visualize_watershed_line,
)


def do_DA_density_refinement(xy_DA_roots, max_nr_of_new_DAs=50,
                             ghp_mode=2, ghp_frame_width=800, ghp_hull_meshsize=500, ghp_shape_simplifier=10,
                             sample_min_DA_distance=150., newDA_prior_target_density=4.0, max_nr_of_tries=1000,
                             write_refinement_steps_to_file=False, show_MCA_ACA_root=False,
                             write_init_final_distribution=False,
                             xy_MCA_root=np.array([]), xy_ACA_root=np.array([]),
                             return_voronoi_polygon_vs_coords = False,
                             filepath_folder=""):
    # Prepare array for new DA coordinates
    xy_DA_roots_new_points = np.array([]).reshape(-1, 2)
    is_initial_DA = np.array([True] * np.size(xy_DA_roots, 0))

    # Voronoi tessalation with DA root coordinates
    # Create ghost points as boundary
    xy_ghost_points = get_ghost_points_for_voronoi_boundary(xy_DA_roots, ghostpoint_mode=ghp_mode,
                                                            frame_width=ghp_frame_width,
                                                            convex_hull_meshsize=ghp_hull_meshsize,
                                                            shape_simplifier=ghp_shape_simplifier)

    vor_initial, is_ghost_pt_initial = do_voronoi_tessalation(xy_DA_roots, use_ghost_points=True,
                                                              xy_ghost_points=xy_ghost_points)

    # vor = vor_initial

    for refinementstep in range(max_nr_of_new_DAs+1):

        vor, is_ghost_pt = do_voronoi_tessalation(xy_DA_roots, use_ghost_points=True,
                                                  xy_ghost_points=xy_ghost_points)  #

        _, area_vor_input_point_based = get_areas_voronoi_polygons(vor)  # area of each polygon: sorted in order of input points

        xy_new_point, is_valid_point = sample_new_DA_point(vor, vor_initial, is_ghost_pt, is_ghost_pt_initial, sample_min_DA_distance,
                                                           newDA_prior_target_density, nr_of_tries=max_nr_of_tries)

        if is_valid_point and refinementstep < max_nr_of_new_DAs:

            _, voronoi_region_ptid_new_pt = find_voronoi_region_for_probepoint(vor, is_ghost_pt, xy_new_point)

            if write_refinement_steps_to_file or (write_init_final_distribution and refinementstep == 0):
                visualize_DAs_refinement(xy_DA_roots, is_initial_DA, vor, is_ghost_pt,
                                         show_areas=True, show_MCA_ACA_root=show_MCA_ACA_root,
                                         xy_MCA_root=xy_MCA_root, xy_ACA_root=xy_ACA_root,
                                         area_vor_input_point_based=area_vor_input_point_based, show_new_point=True,
                                         xy_new_point=xy_new_point,
                                         voronoi_region_ptid_new_pt=voronoi_region_ptid_new_pt,
                                         show_vs_ids=False, title="DAs refinement step " + str(refinementstep),
                                         filepath=filepath_folder + "refinement_nr_" + str(refinementstep) + ".png")

                areas_valid = area_vor_input_point_based[
                    np.logical_and(area_vor_input_point_based > 0, np.logical_not(is_ghost_pt))]
                visualize_DA_distribution(areas_valid,
                                          filepath_density=filepath_folder + "density_distr_refinement_nr_" + str(refinementstep) + ".png",
                                          filepath_area=filepath_folder + "area_distr_refinement_nr_" + str(refinementstep) + ".png")

            xy_DA_roots = np.concatenate((xy_DA_roots, xy_new_point.reshape(-1, 2)), axis=0)
            is_initial_DA = np.append(is_initial_DA, False)
            xy_DA_roots_new_points = np.concatenate((xy_DA_roots_new_points, xy_new_point.reshape(-1, 2)), axis=0)

        else:

            if write_refinement_steps_to_file or write_init_final_distribution:
                visualize_DAs_refinement(xy_DA_roots, is_initial_DA, vor, is_ghost_pt,
                                         show_areas=True, show_MCA_ACA_root=show_MCA_ACA_root,
                                         xy_MCA_root=xy_MCA_root, xy_ACA_root=xy_ACA_root,
                                         area_vor_input_point_based=area_vor_input_point_based, show_new_point=False,
                                         xy_new_point=xy_new_point,
                                         show_vs_ids=False, title="DAs refinement step " + str(refinementstep),
                                         filepath=filepath_folder + "refinement_nr_" + str(refinementstep) + ".png")
                areas_valid = area_vor_input_point_based[
                    np.logical_and(area_vor_input_point_based > 0, np.logical_not(is_ghost_pt))]
                visualize_DA_distribution(areas_valid,
                                          filepath_density=filepath_folder + "density_distr_refinement_nr_" + str(refinementstep) + ".png",
                                          filepath_area=filepath_folder + "area_distr_refinement_nr_" + str(refinementstep) + ".png")

            print("No valid new point found at refinement step", refinementstep)
            break

    if return_voronoi_polygon_vs_coords:
        vor_out, is_ghost_pt_out = do_voronoi_tessalation(xy_DA_roots, use_ghost_points=True,
                                                          xy_ghost_points=xy_ghost_points)
        polygon_vs_xy = []
        for r in range(len(vor_out.point_region)):
            region = vor_out.regions[vor_out.point_region[r]]
            if not -1 in region and not is_ghost_pt_out[r]:
                polygon_vs_xy_current = vor_out.vertices[region]
                polygon_vs_xy.append(polygon_vs_xy_current)
        return xy_DA_roots_new_points, polygon_vs_xy
    else:
        return xy_DA_roots_new_points


def get_ghost_points_for_voronoi_boundary(xy_points, ghostpoint_mode=0, frame_width=800, convex_hull_meshsize=200,
                                          shape_simplifier=10):
    if ghostpoint_mode == 0:
        print("Ghost point mode 0: Do not create any ghost points.")
        xy_ghost_points = np.array([[]]).reshape(-1, 2)

    elif ghostpoint_mode == 1:
        print("Ghost point mode 1: Based on convex hull with constant frame width.")
        print("Frame width =", frame_width)
        print("Meshsize of boundary hull =", convex_hull_meshsize)

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

        print("Ghost point mode 2: Based on the union of circles with midpoint xy_points and radius R.")
        print("Radius =", frame_width)
        print("Resolution simplification parameter =", shape_simplifier)

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
        xy_ghost_points = np.array([[]]).reshape(-1, 2)

    return xy_ghost_points


def do_voronoi_tessalation(xy_DA_roots, use_ghost_points=False, xy_ghost_points=np.array([[]])):
    if use_ghost_points:
        xy_voronoi_pts = np.concatenate((xy_DA_roots, xy_ghost_points), axis=0)
    else:
        xy_voronoi_pts = xy_DA_roots

    is_ghost_pt = np.array([False] * np.size(xy_voronoi_pts, 0))
    is_ghost_pt[np.size(xy_DA_roots, 0):] = True

    vor = Voronoi(xy_voronoi_pts)

    return vor, is_ghost_pt


def get_areas_voronoi_polygons(vor):
    coords_voronoi_vertices = vor.vertices  # coordinates of vertices forming the polygon
    area_vor_regions = np.array([])  # surface area of each polygon

    # find area of each voronoi polygon
    for i in vor.regions:  # i is list of vertices forming each region
        if len(i) > 2 and (-1 not in i):  # if -1: not closed, if only 2 vertices: also not closed
            x_poly = coords_voronoi_vertices[i, 0]
            y_poly = coords_voronoi_vertices[i, 1]
            area_vor_regions = np.append(area_vor_regions, PolyArea(x_poly, y_poly))
        else:
            area_vor_regions = np.append(area_vor_regions, -1)

    nr_of_points = np.size(vor.point_region)
    area_vor_input_point_based = np.zeros(nr_of_points)
    for i in range(nr_of_points):
        current_region = vor.point_region[i]
        area_vor_input_point_based[i] = area_vor_regions[current_region]
    # new criteria
    return area_vor_regions, area_vor_input_point_based  # areas for each polygon: for each region, for each associated point


def is_in_voronoi_region(vor, region, probe_point):
    # check if a certain probe point is within a certain vor region

    if np.size(probe_point) != 2:
        print("Probe point not valid")

    current_region_vs = vor.regions[region]
    xy_current_region_vs = vor.vertices[current_region_vs]

    poly = Polygon(xy_current_region_vs)
    probe_point = Point(probe_point)

    return probe_point.within(poly)


def sample_new_DA_point(vor, vor_initial, is_ghost_point, is_ghost_pt_initial, min_distance, target_density, nr_of_tries=10):
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
        print("New DA coords: ", xy_new_pt, "Nr. of tries:", it, "Is valid DA:", is_valid_point)
        return xy_new_pt, is_valid_point
    else:
        print("No valid DA found after", it, "tries...")
        return np.array([-1.e9, -1.e9]), is_valid_point


def get_closest_distance(xy_current_points, xy_probe_point):
    return np.min(np.linalg.norm(xy_current_points - xy_probe_point, axis=1))


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


# Shoelace formula for area of polygon (https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates)
def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def write_voronoi_polygons_DAs_to_file(polygon_vs_xy, xy_DA_roots_origin, xy_DA_roots_new_points, filepath=""):
    dict_DA_refinement = {}

    polygon_list = []
    for i in polygon_vs_xy:
        polygon_list.append(i.tolist())

    dict_DA_refinement['xy_DA_roots_new_points'] = xy_DA_roots_new_points.tolist()
    dict_DA_refinement['xy_DA_roots_origin'] = xy_DA_roots_origin.tolist()
    dict_DA_refinement['polygon_vs_xy'] = polygon_list

    with open(filepath + 'voronoi_polygons_DA_locs.csv', 'w') as f:
        for key in dict_DA_refinement.keys():
            f.write("%s,%s\n" % (key, dict_DA_refinement[key]))


def add_AVs(polygon_vs_xy, xy_DA_roots, target_da_av_ratio, xy_AV_roots = np.array([]), tol_vs_positions = 1.e-5, min_dist_2_DA = 200, min_dist_2_AV = 200, max_nr_of_tries=1000):

    # param polygon_vs_xy: list with arrays of xy coordinates of all Polygon vertices

    # Identify all unique edges of polygon grid (ignore duplicates)
    is_initial_segment = False
    all_segments = []
    lengths_segments = []
    for current_polygon_xy in polygon_vs_xy:
        if not is_initial_segment:
            all_segments = np.concatenate((current_polygon_xy, np.roll(current_polygon_xy, -1, axis=0)),
                                          axis=1).reshape(-1, 2, 2)
            lengths_segments = np.linalg.norm(current_polygon_xy - np.roll(current_polygon_xy, -1, axis=0), axis=1)
            is_initial_segment = True
        else:
            new_segments = np.concatenate((current_polygon_xy, np.roll(current_polygon_xy, -1, axis=0)),
                                          axis=1).reshape(-1, 2, 2)
            lengths_new_segm = np.linalg.norm(current_polygon_xy - np.roll(current_polygon_xy, -1, axis=0), axis=1)
            for current_segment, current_length in zip(new_segments, lengths_new_segm):
                # identify duplicate new segment which is duplicate to existing segment (if x and y coordinates of
                # both vertices are identical
                # need to consider that the two vertices might be flipped
                nr_of_segments = np.size(all_segments, 0)
                # difference between x and y coordinates of the new segment from all segments. Compute sum of differences,
                # should be zero if points of both vertices coincide.
                sum_of_coord_diffs_segments = np.array(
                    [np.sum(np.abs(all_segments - current_segment)[i]) for i in range(nr_of_segments)])
                # for the case where the two vertices are flipped
                sum_of_coord_diffs_segments_flip = np.array(
                    [np.sum(np.abs(all_segments - np.roll(current_segment, 1, axis=0))[i]) for i in
                     range(nr_of_segments)])
                is_dublicate_segment = np.logical_or(sum_of_coord_diffs_segments < tol_vs_positions,
                                                     sum_of_coord_diffs_segments_flip < tol_vs_positions)
                if True in is_dublicate_segment:
                    continue
                else:
                    # only add if the current segment is a segment which is not already in the all_segments array
                    all_segments = np.concatenate((all_segments, current_segment.reshape(-1, 2, 2)), axis=0)
                    lengths_segments = np.append(lengths_segments, current_length)


    nr_of_new_AVs = target_da_av_ratio * np.size(xy_DA_roots, 0)

    # loop over all new DAs, target ratio, abort criteria

    new_AV_id = 0
    is_valid_AV = False

    for new_AV_id in range(nr_of_new_AVs):

        xy_new_AV_candidate, is_valid_AV = sample_new_AV_point(all_segments,lengths_segments, xy_DA_roots, xy_AV_roots, min_dist_2_DA, min_dist_2_AV, max_nr_of_tries=max_nr_of_tries)

        if is_valid_AV:
            if np.size(xy_AV_roots) > 0:
                xy_AV_roots = np.concatenate((xy_AV_roots, xy_new_AV_candidate.reshape(-1, 2)), axis=0)
            else:
                xy_AV_roots = xy_new_AV_candidate.reshape(-1,2)

        else:
            print("No valid new point found after", new_AV_id, '/', nr_of_new_AVs, 'added')
            break

    if is_valid_AV:
        print("Total of", new_AV_id+1, '/', nr_of_new_AVs, 'added')
    return xy_AV_roots


def sample_new_AV_point(all_segments, lengths_segments, xy_DA_roots, xy_AV_roots, min_dist_2_DA, min_dist_2_AV, max_nr_of_tries = 10):

    # Individual and cumulative sum of all segment lengths, normalized with total length of all segments
    rel_lengths_segments = lengths_segments / np.sum(lengths_segments)
    rel_lengths_cumulati = np.cumsum(lengths_segments) / np.sum(lengths_segments)

    is_valid_point = False
    xy_new_AV_candidate = np.array([-1.e9, -1.e9])
    it = 0

    while (not is_valid_point) and it < max_nr_of_tries:

        it += 1

        # sample new location with uniform distribution
        sample_loc = np.random.rand()  # uniformly distributed random number
        sample_segment_index = np.where(rel_lengths_cumulati > sample_loc)[0][0]

        rel_loc_up = rel_lengths_cumulati[sample_segment_index]
        rel_loc_delta = rel_lengths_segments[sample_segment_index]
        rel_loc_low = rel_loc_up - rel_loc_delta

        xy_new_AV_candidate = all_segments[sample_segment_index][0, :] + (sample_loc - rel_loc_low) / rel_loc_delta * (
                all_segments[sample_segment_index][1, :] - all_segments[sample_segment_index][0, :])

        if np.size(xy_DA_roots,axis=0) > 0:
            closest_distance_2_DAs = get_closest_distance(xy_DA_roots, xy_new_AV_candidate)
        else:
            closest_distance_2_DAs = np.sum(lengths_segments) * 100 # set a very large value, such that always True

        if np.size(xy_AV_roots,axis=0) > 0:
            closest_distance_2_AVs = get_closest_distance(xy_AV_roots, xy_new_AV_candidate)
        else:
            closest_distance_2_AVs = np.sum(lengths_segments) * 100 # set a very large value, such that always True

        if closest_distance_2_DAs > min_dist_2_DA and closest_distance_2_AVs > min_dist_2_AV:
            # accept
            is_valid_point = True
        else:
            continue

    if is_valid_point:
        print("New AV coords: ", xy_new_AV_candidate, "Nr. of tries:", it, "Is valid AV:", is_valid_point)
        return xy_new_AV_candidate, is_valid_point
    else:
        print("No valid AV found after", it, "tries...")
        return np.array([-1.e9, -1.e9]), is_valid_point


def write_AV_locs_to_file(xy_AVs, filepath=""):

    dict_AV = {}
    dict_AV['xy_AVs'] = xy_AVs.tolist()

    with open(filepath + 'AV_locs.csv', 'w') as f:
        for key in dict_AV.keys():
            f.write("%s,%s\n" % (key, dict_AV[key]))
