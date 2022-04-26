from trame import setup_dev

from . import controller, ui


def start_server():
    setup_dev(ui, controller)
    print("dev_setup")
    ui.layout.start()
    print("layout started")
    update_plot("MeanStd")
    print("server started")


def start_desktop():
    ui.layout.start_desktop_window()


def main():
    controller.on_start()
    print("on start")
    start_server()
    print("all done")


if __name__ == "__main__":
    main()
