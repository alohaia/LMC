import igraph
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns


def visualize_DAs_refinement(xy_DA_roots, is_initial_DA, vor, is_ghost_pt, show_areas=False,
                             show_MCA_ACA_root = False, xy_MCA_root = np.array([]), xy_ACA_root = np.array([]),
                             area_vor_input_point_based=np.array([]), show_new_point=False, xy_new_point=np.array([]),
                             voronoi_region_ptid_new_pt = np.array([]), show_vs_ids=False, title="Test",
                             filepath="newfile.png"):

    fig, ax = plt.subplots()

    ax.plot(xy_DA_roots[is_initial_DA, 0], xy_DA_roots[is_initial_DA, 1], 'o', color='#f21e07', zorder=1, label='DA root', markersize=5)
    ax.plot(xy_DA_roots[np.logical_not(is_initial_DA), 0], xy_DA_roots[np.logical_not(is_initial_DA), 1], 'o',
            color='#e1a9a3', zorder=1, label='DA root added', markersize=5)

    if show_MCA_ACA_root and np.size(xy_MCA_root) == 2 and np.size(xy_ACA_root) == 2:
        ax.plot(xy_MCA_root[:, 0], xy_MCA_root[:, 1], 'o', color='#7ea5e0', markersize=9, zorder=1,
                label='MCA in')  # MCA
        ax.plot(xy_ACA_root[:, 0], xy_ACA_root[:, 1], 'o', color='#97e07e', markersize=9, zorder=1,
                label='ACA in')  # ACA

    # Plot voronoi polygons, only closed ones, and polygons that do not involve ghost points
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region and not is_ghost_pt[r]:
            patch = patches.Polygon(vor.vertices[region], closed=True, fill=False)
            ax.add_patch(patch)

    ax.plot(vor.points[is_ghost_pt, 0], vor.points[is_ghost_pt, 1], 'o', color='g', markersize=1.5,
            zorder=1,
            label='Ghost point')

    if show_new_point and np.size(xy_new_point) == 2:
        ax.plot(xy_new_point[0], xy_new_point[1], 'xg', label = 'New valid point')

        if voronoi_region_ptid_new_pt >= 0:
            region = vor.regions[vor.point_region[voronoi_region_ptid_new_pt]]
            patch_fill = patches.Polygon(vor.vertices[region], closed=True, fill=True, alpha=.5)
            ax.add_patch(patch_fill)

    if show_areas:
        for j in range(np.size(vor.points, 0)):
            current_xy = vor.points[j, :]
            current_area = area_vor_input_point_based[j]
            if current_area > 0. and is_ghost_pt[j] == False:
                ax.text(current_xy[0], current_xy[1], "{:.2E}".format(current_area), fontsize=3,
                        horizontalalignment='center', verticalalignment='top')

    ax.legend(prop={'size': 9})
    ax.set_title(title)

    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')

    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    fig.savefig(filepath, dpi=600)

    plt.close(fig)

    return


def visualize_DA_distribution(areas, density_ref_surface = 1.e6, filepath_density ="distplot.png", filepath_area ="distplot.png"):

    denities_estimated = density_ref_surface / areas

    nr_of_DA_areas = np.size(areas)
    total_area = np.sum(areas)
    overall_DA_density = nr_of_DA_areas / total_area * density_ref_surface

    df_voronoi = pd.DataFrame(data={'Areas_init': areas, 'Densities_initial': denities_estimated})

    fig2, ax2 = plt.subplots()

    sns.distplot(df_voronoi['Densities_initial'], ax=ax2,rug=True, hist=False)

    ax2.axvline(df_voronoi['Densities_initial'].mean(), color='0', linestyle='-', label='Mean')
    ax2.axvline(df_voronoi['Densities_initial'].median(), color='.5', linestyle='-', label='Median')
    ax2.axvline(df_voronoi['Densities_initial'].quantile(.25), color='.5', linestyle='--', label='0.25 Quantile')
    ax2.axvline(df_voronoi['Densities_initial'].quantile(.75), color='.5', linestyle=':', label='0.75 Quantile')
    ax2.axvline(overall_DA_density, color='#FF0040', linestyle='-', label='Overall DA density')

    ax2.legend()

    fig2.savefig(filepath_density, dpi=600)

    plt.close(fig2)


    fig3, ax3 = plt.subplots()

    sns.distplot(df_voronoi['Areas_init'], ax=ax3,rug=True, hist=False)

    ax3.axvline(df_voronoi['Areas_init'].mean(), color='0', linestyle='-', label='Mean')
    ax3.axvline(df_voronoi['Areas_init'].median(), color='.5', linestyle='-', label='Median')
    ax3.axvline(df_voronoi['Areas_init'].quantile(.25), color='.5', linestyle='--', label='0.25 Quantile')
    ax3.axvline(df_voronoi['Areas_init'].quantile(.75), color='.5', linestyle=':', label='0.75 Quantile')
    # ax3.axvline(overall_DA_density, color='#FF0040', linestyle='-', label='Overall DA density')

    ax3.legend()

    fig3.savefig(filepath_area, dpi=600)

    plt.close(fig3)

    return


def visualize_watershed_line(xy_DAroots, is_DAroot_MCA, is_DAroot_ACA, vor, is_ghost_pt, segments_watershed_line,
                             vid_MCA_DAs_at_watershedline, vid_ACA_DAs_at_watershedline, graph, filepath='tmp.png'):

    fig, ax = plt.subplots()

    ax.plot(xy_DAroots[is_DAroot_MCA, 0], xy_DAroots[is_DAroot_MCA, 1], 'o', color='#7ea5e0', zorder=1,
            label='MCA DA root', markersize=5)
    ax.plot(xy_DAroots[is_DAroot_ACA, 0], xy_DAroots[is_DAroot_ACA, 1], 'o', color='#97e07e', zorder=1,
            label='ACA DA root', markersize=5)

    ax.plot(np.array(graph.vs["coords"])[vid_MCA_DAs_at_watershedline, 0],
            np.array(graph.vs["coords"])[vid_MCA_DAs_at_watershedline, 1], 'o', color='None', zorder=2,
            markeredgecolor = '.6', markerfacecolor = 'None', markersize=5)

    ax.plot(np.array(graph.vs["coords"])[vid_ACA_DAs_at_watershedline, 0],
            np.array(graph.vs["coords"])[vid_ACA_DAs_at_watershedline, 1], 'o', color='None', zorder=2,
            markeredgecolor = '.6', markerfacecolor = 'None', markersize=5)

    # Plot voronoi polygons, only closed ones, and polygons that do not involve ghost points
    for r in range(len(is_DAroot_MCA)): # vor_points from DAroots are first entries in vor (before ghost_pts)
        region = vor.regions[vor.point_region[r]]
        if not -1 in region and not is_ghost_pt[r]:
            patch = patches.Polygon(vor.vertices[region], closed=True, fill=False)
            ax.add_patch(patch)
            if is_DAroot_ACA[
                r]:
                patch_fill = patches.Polygon(vor.vertices[region], closed=True, fill=True, color='#97e07e', alpha=.5,
                                             zorder=0)
                ax.add_patch(patch_fill)
            elif is_DAroot_MCA[r]:
                patch_fill = patches.Polygon(vor.vertices[region], closed=True, fill=True, color='#7ea5e0', alpha=.5,
                                             zorder=0)
                ax.add_patch(patch_fill)

    ax.plot(vor.points[is_ghost_pt, 0], vor.points[is_ghost_pt, 1], 'o', color='g', markersize=1.5,
            zorder=1, label='Ghost point')

    # vessel segments (PAs incl. collaterals)
    is_PA = np.array(graph.es['type']) <= 0

    adjacency_list = np.array(graph.get_edgelist(), dtype=int)[is_PA]
    x = np.array(graph.vs["coords"])[:, 0]
    y = np.array(graph.vs["coords"])[:, 1]

    tmp = -np.ones(
        (np.size(adjacency_list, 0), 4))  # arange coordinates in tmp array which will be used for segments later
    tmp[:, 0] = x[adjacency_list[:, 0]]
    tmp[:, 1] = y[adjacency_list[:, 0]]
    tmp[:, 2] = x[adjacency_list[:, 1]]
    tmp[:, 3] = y[adjacency_list[:, 1]]

    segments = tmp.reshape(-1, 2, 2)

    from matplotlib.collections import LineCollection

    lc = LineCollection(segments, colors='.2')

    ax.add_collection(lc)

    lc_watershed = LineCollection(segments_watershed_line, colors='red')
    ax.add_collection(lc_watershed)

    ax.legend(prop={'size': 9})
    # ax.set_title(title)

    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')

    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    fig.savefig(filepath, dpi=600)

    plt.close(fig)


def visualize_DA_AV_roots_with_polygons(xy_DA_roots, is_initial_DA, xy_AV_roots,
                                        polygon_vs_xy, show_MCA_ACA_root=False,
                                        xy_MCA_root=np.array([]), xy_ACA_root=np.array([]),
                                        title = "Title", filepath="newfile.png"):
    fig, ax = plt.subplots()

    ax.plot(xy_DA_roots[is_initial_DA, 0], xy_DA_roots[is_initial_DA, 1], 'o', color='#f21e07', zorder=1,
            label='DA root', markersize=5)
    ax.plot(xy_DA_roots[np.logical_not(is_initial_DA), 0], xy_DA_roots[np.logical_not(is_initial_DA), 1], 'o',
            color='#e1a9a3', zorder=1, label='DA root added', markersize=5)

    ax.plot(xy_AV_roots[:, 0], xy_AV_roots[:, 1], 'o', color='#0b8292', zorder=1, label='AV root added', markersize=4)

    if show_MCA_ACA_root and np.size(xy_MCA_root) == 2 and np.size(xy_ACA_root) == 2:
        ax.plot(xy_MCA_root[:, 0], xy_MCA_root[:, 1], 'o', color='#7ea5e0', markersize=9, zorder=1,
                label='MCA in')  # MCA
        ax.plot(xy_ACA_root[:, 0], xy_ACA_root[:, 1], 'o', color='#97e07e', markersize=9, zorder=1,
                label='ACA in')  # ACA

    # Plot voronoi polygons, only closed ones, and polygons that do not involve ghost points
    for r in range(len(polygon_vs_xy)):
        patch = patches.Polygon(polygon_vs_xy[r], closed=True, fill=False)
        ax.add_patch(patch)

    ax.legend(prop={'size': 9})
    ax.set_title(title)

    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')

    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    fig.savefig(filepath, dpi=600)

    plt.close(fig)

    return


def plot_pial_vasculature(x, y, adjacency_list, diameters, is_collateral, is_added_manually, is_DA_startingPt,
                          is_DA_startingPt_added_manually, is_MCA_in, is_ACA_in, xy_AV_roots=np.array([]),
                          show_vs_ids=False, title="Test", filepath="test.png"):

    nr_of_edges = np.size(diameters)

    fig, ax = plt.subplots()

    xy_DA_roots = np.array([x[is_DA_startingPt], y[is_DA_startingPt]]).transpose()
    xy_DA_roots_added = np.array([x[is_DA_startingPt_added_manually], y[is_DA_startingPt_added_manually]]).transpose()

    ax.plot(xy_DA_roots[:, 0], xy_DA_roots[:, 1], 'o', color='#f21e07', zorder=1, label='DA root', markersize=5)
    ax.plot(xy_DA_roots_added[:, 0], xy_DA_roots_added[:, 1], 'o', color='#e1a9a3', zorder=1, label='DA root added',markersize=5)

    if np.size(xy_AV_roots) > 1:
        ax.plot(xy_AV_roots[:, 0], xy_AV_roots[:, 1], 'o', color='#0b8292', zorder=1, label='AV root added', markersize=4)

    # MCA / ACA inlet
    xy_MCA_root = np.array([x[is_MCA_in], y[is_MCA_in]]).transpose()
    xy_ACA_root = np.array([x[is_ACA_in], y[is_ACA_in]]).transpose()

    ax.plot(xy_MCA_root[:, 0], xy_MCA_root[:, 1], 'o', color='#7ea5e0', markersize=9, zorder=1, label='MCA in') # MCA
    ax.plot(xy_ACA_root[:, 0], xy_ACA_root[:, 1], 'o', color='#97e07e', markersize=9, zorder=1, label='ACA in') # ACA

    # vessel segments (PAs incl. collaterals)
    tmp = -np.ones((nr_of_edges, 4))  # arange coordinates in tmp array which will be used for segments later
    tmp[:, 0] = x[adjacency_list[:, 0]]
    tmp[:, 1] = y[adjacency_list[:, 0]]
    tmp[:, 2] = x[adjacency_list[:, 1]]
    tmp[:, 3] = y[adjacency_list[:, 1]]

    segments = tmp.reshape(-1, 2, 2)

    linecolors = np.array(["#f21e07"] * nr_of_edges)
    linecolors[is_added_manually] = "0.7"
    linecolors[is_collateral] = "#e0d97e"
    lc = LineCollection(segments, colors=linecolors)

    linewidth_vessel = np.copy(diameters) * .1
    lc.set_linewidth(linewidth_vessel)
    ax.add_collection(lc)

    if show_vs_ids:
        nr_of_vs = np.size(x)
        for vs_id in np.arange(nr_of_vs):
            ax.text(x[vs_id], y[vs_id], "{:.0f}".format(vs_id), fontsize=5,
                    horizontalalignment='center', verticalalignment='bottom')

    # ax.set_xlim(-2000,4000)
    # ax.set_ylim(-2500, 2000)

    ax.legend(prop={'size': 9})
    ax.set_title(title)

    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')

    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    fig.savefig(filepath, dpi=600)

    plt.close(fig)
