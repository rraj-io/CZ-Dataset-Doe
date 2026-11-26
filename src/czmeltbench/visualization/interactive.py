"""
File Description: Interactivve PyVista CFD Viever fro CZMeltBench
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-25
Version: 1.0
"""

from typing import Literal, TypeAlias

import pyvista as pv

from czmeltbench.visualization.vtk import (
    SimulationReader,
    create_vtk_grid,
)

FieldName: TypeAlias = Literal[
    "u_r",
    "u_z",
    "u_swirl",
    "p",
    "T",
    "meridional_speed",
    "total_speed",
]

VIEW_FIELDS: tuple[str, ...] = (
    "T",
    "u_r",
    "u_z",
    "u_swirl",
    "p",
    "meridional_speed",
    "total_speed",
)

def show_case_interactive(reader: SimulationReader,
                          case_id: str,
                          *,
                          initial_field: FieldName = "T",
                          show_edges: bool  = False,
                          cmap: str = "viridis",
                          ) -> pv.Plotter:
    """Open an interactive native-mesh CFD viewer

    Parameters
    ----------
    reader : SimulationReader
        Reader for CFD simulaiton data.

    case_id : str
        simulation identifier.

    initial_field : Fieldname, default="T"
        Field shown on startup.

    show_edges : bool, default=False
        Whether to show edges initially.

    cmap : str, default="viridis
        matplotlib compatible colormap name.

    Returns
    -------
    pv.Plotter
        Interactive PyVista plotter.

    Controls
    ---------
    n : Next field.
    p : previous field.
    m : Toggle mesh-only / CFD view.
    e : Toggle cell edges.
    """

    if initial_field not in VIEW_FIELDS:
        raise ValueError(f"Unknown field: {initial_field}")

    pos, _, y = reader.load_arrays(case_id)

    grid: pv.UnstructuredGrid = create_vtk_grid(pos=pos,
                                                y=y)

    plotter: pv.Plotter = pv.Plotter(
        window_size=(1100, 750),
    )

    plotter.set_background("white")

    field_index: int = VIEW_FIELDS.index(initial_field)
    mesh_only: bool = False
    edges_visible: bool = show_edges

    def update_view() -> None:
        """Refresh the surface actor and status information.
        """
        field: str = VIEW_FIELDS[field_index]

        # Remove the previous field's scalar bar
        for title in list(plotter.scalar_bars.keys()):
            plotter.remove_scalar_bar(title, render=False)

        # Remove previous mesh actor
        plotter.remove_actor("cz_surface", render=False)

        if mesh_only:
            plotter.add_mesh(grid,
                             name="cz_surface",
                             color="lightgrey",
                             style="wireframe",
                             line_width=1.0,
                             show_scalar_bar=False,
                             reset_camera=False,
                             )
            mode_text: str = "Mesh only"
        else:
            plotter.add_mesh(grid,
                             name="cz_surface",
                             scalars=field,
                             cmap=cmap,
                             show_edges=edges_visible,
                             edge_color="black",
                             line_width=0.6,
                             show_scalar_bar=True,
                             scalar_bar_args={"title": field},
                             reset_camera=False,
                             )
            mode_text = "CFD fields"

        plotter.add_text(
            f"{case_id} | {mode_text} | Field: {field}",
            position="upper_left",
            font_size=11,
            color="black",
            name="status_text",
        )

        plotter.render()


    def next_field() -> None:
        """Switch to the next availabel field.
        """

        nonlocal field_index, mesh_only
        field_index = (field_index + 1) % len(VIEW_FIELDS)
        mesh_only = False
        update_view()


    def previous_field() -> None:
        """Switch to the previous available field.
        """

        nonlocal field_index, mesh_only
        field_index = (field_index - 1) % len(VIEW_FIELDS)
        mesh_only = False
        update_view()

    def toggle_mesh() -> None:
        """Switch between mesh-only and CFD views.
        """
        nonlocal mesh_only
        mesh_only = not mesh_only
        update_view()

    def toggle_edges() -> None:
        """Toggle cell edges in CFD view.
        """
        nonlocal edges_visible
        edges_visible = not edges_visible
        update_view()


    plotter.add_key_event("n", next_field)
    plotter.add_key_event("p", previous_field)
    plotter.add_key_event("m", toggle_mesh)
    plotter.add_key_event("e", toggle_edges)


    plotter.add_axes()

    # Display the meridional r-z plane
    plotter.view_xy()
    plotter.camera.parallel_projection = True

    update_view()

    plotter.add_text(
        "n/p: change field | m: mesh | e: edges",
        position="lower_left",
        font_size=10,
        color="black",
        name="help_text",
    )

    plotter.show()

    return plotter

        