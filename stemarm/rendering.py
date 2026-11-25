"""Exudyn windows for single-process simulations."""
from contextlib import contextmanager

import exudyn as exu
import numpy as np


@contextmanager
def live(sc: exu.SystemContainer, enabled: bool, *, title: str = "", pause: bool = False,
         rotation: np.ndarray | None = None):
    """Display a solver run and always close its renderer on exit."""
    if not enabled:
        yield
        return

    view = sc.visualizationSettings
    view.general.autoFitScene = True
    view.general.renderWindowString = title
    view.window.renderWindowSize = [1200, 800]
    view.nodes.show = False
    view.markers.show = False
    view.loads.show = False
    view.bodies.beams.axialTiling = 12
    renderer = sc.renderer
    if not renderer.Start():
        raise RuntimeError("Could not open the Exudyn OpenGL renderer.")
    try:
        if rotation is not None:
            state = renderer.GetState()
            state["modelRotation"] = rotation
            renderer.SetState(state)
        renderer.ZoomAll()
        renderer.DoIdleTasks(waitSeconds=-1.0 if pause else 0.25, printPauseMessage=pause)
        yield
        if renderer.IsActive():
            renderer.DoIdleTasks(waitSeconds=-1.0 if pause else 1.0,
                                 printPauseMessage=pause)
    finally:
        renderer.Stop()
