from trame.app import get_server
from .core import Engine
from . import ui, state_manager, pce_builder


def main(server=None, **kwargs):
    # Get or create server
    if server is None:
        server = get_server()

    if isinstance(server, str):
        server = get_server(server)

    # Init application
    print("staring initialization")
    state_manager.initialize(server)
    print("state initializated")
    pce_builder.initialize(server)
    print("builder initializated")
    Engine(server)
    print("engine initializated")
    ui.initialize(server)
    print("ui initializated")
    server.controller.reset_all()

    print("initializated and reset")

    # Start server
    server.start(**kwargs)


if __name__ == "__main__":
    main()
