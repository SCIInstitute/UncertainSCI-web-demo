import multiprocessing

from uncertainsci_web_demo.app.main import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main(exec_mode="desktop")
