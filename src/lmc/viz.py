from igraph import Edge, Graph
from matplotlib import colors
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.collections import LineCollection

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from scipy.spatial import Voronoi, voronoi_plot_2d

from lmc.core import ops, io
from lmc.types import *
from lmc.config import palette

def plot_sa(graph_sa: Graph) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    vs_sa = ops.get_vs(graph_sa, z=False)
    ax.add_collection(LineCollection(
        vs_sa[[(e.source, e.target) for e in graph_sa.es(origin='MCA')]].tolist(),
        colors="red", linewidths=4
    ))
    for v in graph_sa.vs(type_in=('DA root', 'DA root added manually')):
        ax.text(v['x'], v['y'], f'{v.index}|{v['name']:.0f}', fontsize=6, color='black')
    for v in graph_sa.vs(type_in=('PA point')):
        ax.text(v['x'], v['y'], f'{v.index}|{v['name']:.0f}', fontsize=4, color='gray')
    ax.add_collection(LineCollection(
        vs_sa[[(e.source, e.target) for e in graph_sa.es(origin='ACA')]].tolist(),
        colors="blue", linewidths=4
    ))
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    return fig, ax

def plot_graph(g: Graph) -> tuple[Figure, Axes]:
    """Visualize a graph."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Vertices and points {{{
    vs_da = ops.filter_vs(g, vtype=("DA root",), z=False)
    vs_da_added = ops.filter_vs(g, vtype=("DA root added manually",), z=False)
    vs_av = ops.filter_vs(g, vtype=("AV root",), z=False)
    ax.scatter(
        label="Original DA root",
        x=vs_da[:, 0], y=vs_da[:, 1],
        s=150, marker="o", edgecolors="black", c=palette["DA root"],
    )
    ax.scatter(
        label="Sampled DA root",
        x=vs_da_added[:, 0], y=vs_da_added[:, 1],
        s=150, marker="o", edgecolors="black", c=palette["DA root added manually"],
    )
    ax.scatter(
        label="Sampled AV root",
        x=vs_av[:, 0], y=vs_av[:, 1],
        s=50, marker="o", edgecolors="black", c=palette["AV root"],
    )
    # }}}

    # Edges and ridgets {{{
    es_pa = ops.filter_es(g, attr_not=("is_added_manually", "is_collateral"),
                          etype="PA", z=False)
    es_pa_added = ops.filter_es(g, attr="is_added_manually",
                                attr_not="is_collateral", etype="PA", z=False)
    es_pa_col = ops.filter_es(g, attr="is_collateral", etype="PA", z=False)

    ax.add_collection(LineCollection(es_pa.tolist(), colors="#810400", linewidths=4))
    ax.add_collection(LineCollection(es_pa_added.tolist(), colors="#FE9A56", linewidths=4))
    ax.add_collection(LineCollection(es_pa_col.tolist(), colors="#E0DA7F", linewidths=4, label="LMC"))
    # }}}

    # Scale bar {{{
    scalebar_length = 500
    margin = 50
    x0 = ax.get_xlim()[1] - scalebar_length
    y0 = ax.get_ylim()[0] - margin
    ax.hlines(y=y0, xmin=x0, xmax=x0 + scalebar_length, colors='k', linewidth=2)
    ax.text(x0 + scalebar_length / 2, y0 - margin/2, f'{scalebar_length} μm',
        ha='center', va='bottom', color='k', fontsize=10)
    # }}}

    fig.subplots_adjust(bottom=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0), fontsize=10)
    ax.grid(False)
    ax.axis('off')
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    return fig, ax

def plot_caps(g):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    vertex_type_to_color = {
        "p": "black", "q": "black",
        "a": "red",
        "b": "green",
        "c": "blue",
        "d1": "yellow", "d2": "yellow",
        "split node": "gray",
    }
    vertex_colors = [vertex_type_to_color[t] for t in g.vs["type"]]

    edge_type_to_color = {
        "e1": "black", "e2": "black", "e3": "black",
        "a": "red",
        "b": "green",
        "c": "blue",
        "d1": "yellow", "d2": "yellow",
        "connect edge": "gray"
    }
    edge_colors = [edge_type_to_color[t] for t in g.es["type"]]

    ax.scatter(
        g.vs["x"], g.vs["y"],
        # label=vs_type,
        c=vertex_colors,
        s=20,
        zorder=3
    )

    ax.add_collection(LineCollection(
        ops.get_es(g, z=False).tolist(),
        # label=es_type,
        colors=edge_colors,
        linewidths=1.5,
        alpha=0.8
    ))

    # ax.legend()
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    return fig, ax

# def plot_edges(edges, ax, **kwargs):
#     lc = LineCollection(edges, **kwargs)
#     ax.add_collection(lc)
#
#     return ax

def plot_voronoi(
    vor: Voronoi,
    title: str="",
    show_points: bool=True,
    show_vertices: bool=True,
) -> tuple[Figure, Axes]:
    """Visualize a Voronoi tessalation."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # plot edge, region and input points
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray',
                    line_width=1, line_alpha=0.6)

    # redraw input points
    if show_points:
        ax.plot(vor.points[:, 0], vor.points[:, 1], 'o', color='blue',
                markersize=8, label='Input Points (vor.points)')
        # Add tags for each point
        for i, p in enumerate(vor.points):
            ax.text(p[0] + 0.1, p[1] + 0.1, f'P{i}', color='blue', fontsize=10)

    # redraw computed vertices
    if show_vertices:
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'x', color='red',
                markersize=10, label='Voronoi Vertices (vor.vertices)')
        # Add tags for each vertex
        for i, v in enumerate(vor.vertices):
            ax.text(v[0] + 0.1, v[1] + 0.1, f'V{i}', color='red', fontsize=10)


    ax.set_title(title)
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    return fig, ax


def plot_DAs_refinement(
    graph: Graph,
    xys_DA_roots_added: Vertices,
    vor: VoronoiExt,
    area_vor_input_point_based: NDArray[np.float64],
    xy_new_point: Vertex,
    i_voronoi_newpt_corr_regionpt: np.intp,
    show_new_point: bool=False,
    show_areas: bool=False,
    show_MCA_ACA_root: bool=False,
    show_vs_ids: bool=False,
    title="Test",
    filepath="newfile.png"
) -> None:
    xys_DA_roots = ops.filter_vs(graph, attr="is_DA_root", z=False)
    xys_MCA_roots = ops.filter_vs(graph, attr="is_MCA_in", z=False)
    xys_ACA_roots = ops.filter_vs(graph, attr="is_ACA_in", z=False)
    # xys_ghost_points = ops.filter_vs(graph, vtype=("Ghost point",), z=False)

    fig, ax = plt.subplots()

    ax.plot(
        xys_DA_roots[:, 0], xys_DA_roots[:, 1], 'o',
        color=palette["DA root"], zorder=1, label='DA root', markersize=5)
    ax.plot(
        xys_DA_roots_added[:, 0], xys_DA_roots_added[:, 1], 'o',
        color=palette["DA root added manually"], zorder=1, label='DA root added',
        markersize=5)

    if show_MCA_ACA_root and \
        np.size(xys_MCA_roots) == 2 and np.size(xys_ACA_roots) == 2:

        ax.plot(xys_MCA_roots[:, 0], xys_MCA_roots[:, 1], 'o',
                color=palette["MCA inflow"], zorder=1, label='MCA in',
                markersize=9)
        ax.plot(xys_ACA_roots[:, 0], xys_ACA_roots[:, 1], 'o',
                color=palette["ACA inflow"], zorder=1, label='ACA in',
                markersize=9)

    # Plot voronoi polygons, only closed ones, and polygons that do not involve ghost points
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region and vor.point_annot[r] == "DA root":
            patch = patches.Polygon(vor.vertices[region], closed=True,
                                    fill=False)
            ax.add_patch(patch)

    ax.plot(vor.points[vor.point_annot == "Ghost point", 0],
            vor.points[vor.point_annot == "Ghost point", 1],
            'o', color='g', markersize=1.5, zorder=1, label='Ghost point')

    if show_new_point and np.size(xy_new_point) == 2:
        ax.plot(xy_new_point[0], xy_new_point[1], 'xg', label = 'New valid point')

        if i_voronoi_newpt_corr_regionpt >= 0:
            region = vor.regions[vor.point_region[i_voronoi_newpt_corr_regionpt]]
            patch_fill = patches.Polygon(vor.vertices[region], closed=True, fill=True, alpha=.5)
            ax.add_patch(patch_fill)

    if show_areas:
        for j in range(np.size(vor.points, 0)):
            current_xy = vor.points[j, :]
            current_area = area_vor_input_point_based[j]
            if current_area > 0. and vor.point_annot[j] == "Ghost point":
                ax.text(current_xy[0], current_xy[1], "{:.2E}".format(current_area), fontsize=3,
                        horizontalalignment='center', verticalalignment='top')

    ax.legend(prop={'size': 9})
    ax.set_title(title)

    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel('x [μm]')
    ax.set_ylabel('y [μm]')

    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    fig.savefig(filepath, dpi=600)

    plt.close(fig)


def plot_DA_distribution(
    areas,
    density_ref_surface = 1.e6,
    filepath_density ="distplot.png",
    filepath_area ="distplot.png"
):

    pass

    # denities_estimated = density_ref_surface / areas
    #
    # nr_of_DA_areas = np.size(areas)
    # total_area = np.sum(areas)
    # overall_DA_density = nr_of_DA_areas / total_area * density_ref_surface
    #
    # df_voronoi = pd.DataFrame({
    #     'Areas_init': areas,
    #     'Densities_initial': denities_estimated
    # })
    #
    # fig2, ax2 = plt.subplots()
    #
    # sns.histplot(df_voronoi['Densities_initial'], ax=ax2,rug=True, hist=False)
    #
    # ax2.axvline(df_voronoi['Densities_initial'].mean(),
    #             color='0', linestyle='-', label='Mean')
    # ax2.axvline(df_voronoi['Densities_initial'].median(),
    #             color='.5', linestyle='-', label='Median')
    # ax2.axvline(df_voronoi['Densities_initial'].quantile(.25),
    #             color='.5', linestyle='--', label='0.25 Quantile')
    # ax2.axvline(df_voronoi['Densities_initial'].quantile(.75),
    #             color='.5', linestyle=':', label='0.75 Quantile')
    # ax2.axvline(overall_DA_density,
    #             color='#FF0040', linestyle='-', label='Overall DA density')
    #
    # ax2.legend()
    #
    # fig2.savefig(filepath_density, dpi=600)
    #
    # plt.close(fig2)
    #
    #
    # fig3, ax3 = plt.subplots()
    #
    # sns.histplot(df_voronoi['Areas_init'], ax=ax3,rug=True, hist=False)
    #
    # ax3.axvline(df_voronoi['Areas_init'].mean(), color='0', linestyle='-', label='Mean')
    # ax3.axvline(df_voronoi['Areas_init'].median(), color='.5', linestyle='-', label='Median')
    # ax3.axvline(df_voronoi['Areas_init'].quantile(.25), color='.5', linestyle='--', label='0.25 Quantile')
    # ax3.axvline(df_voronoi['Areas_init'].quantile(.75), color='.5', linestyle=':', label='0.75 Quantile')
    # # ax3.axvline(overall_DA_density, color='#FF0040', linestyle='-', label='Overall DA density')
    #
    # ax3.legend()
    #
    # fig3.savefig(filepath_area, dpi=600)
    #
    # plt.close(fig3)
    #
    # return


# TODO
def plot_DA_AV_roots_with_polygons(
    graph: Graph,
    xys_new_DA_roots: Vertices,
    xy_AV_roots: Vertices,
    polygon_vs_xy,
    show_MCA_ACA_root=False,
    title = "Title",
    filepath="newfile.png"
) -> None:
    fig, ax = plt.subplots()

    xys_vs = np.array(graph.vs["coords"])[:, 0:2]
    xys_DA_roots = xys_vs[graph.vs["is_DA_root"], ]
    xys_MCA_roots = xys_vs[graph.vs["is_MCA_in"], ]
    xys_ACA_roots = xys_vs[graph.vs["is_ACA_in"], ]

    ax.plot(xys_DA_roots[:, 0], xys_DA_roots[:, 1], "o",
            color=palette["DA root"], zorder=1, label="DA root", markersize=5)
    ax.plot(xys_new_DA_roots[:, 0], xys_DA_roots[:, 1], "o",
            color=palette["DA root added"], zorder=1, label="DA root added",
            markersize=5)

    ax.plot(xy_AV_roots[:, 0], xy_AV_roots[:, 1], 'o', color='#0b8292', zorder=1, label='AV root added', markersize=4)

    if show_MCA_ACA_root and np.size(xys_MCA_roots) == 2 and np.size(xys_ACA_roots) == 2:
        ax.plot(xys_MCA_roots[:, 0], xys_MCA_roots[:, 1], 'o', color='#7ea5e0', markersize=9, zorder=1,
                label='MCA in')  # MCA
        ax.plot(xys_ACA_roots[:, 0], xys_ACA_roots[:, 1], 'o', color='#97e07e', markersize=9, zorder=1,
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

    ax.set_xlabel('x [μm]')
    ax.set_ylabel('y [μm]')

    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    fig.savefig(filepath, dpi=600)

    plt.close(fig)

    return
