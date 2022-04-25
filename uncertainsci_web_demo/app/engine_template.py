r"""
Define your classes and create the instances that you need to expose
"""
from trame import state

# ---------------------------------------------------------
# Methods
# ---------------------------------------------------------

def initialize():
    print(">>> ENGINE: Application initialize only once")

def protocols_ready():
    print(">>> ENGINE: Server protocols initialized / Client not connected yet")

def reset_resolution():
    state.resolution = 6
def widget_click():
    print(">>> ENGINE: Widget Click")

def widget_change():
    print(">>> ENGINE: Widget Change")

# ---------------------------------------------------------
# Listeners
# ---------------------------------------------------------

@state.change("resolution")
def resolution_changed(resolution, **kwargs):
    print(f">>> ENGINE: Slider updating resolution to {resolution}")
