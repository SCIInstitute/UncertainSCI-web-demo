from setuptools import setup

ENTRY_POINT = ["uncertainsci-web-demo"]
DATA_FILES = []

OPTIONS = {
    "argv_emulation": False,
    "strip": True,
    "iconfile": "uncertainsci-web-demo.icns",
    "includes": ["WebKit", "Foundation", "setuptools"],
}

setup(
    app=ENTRY_POINT,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
