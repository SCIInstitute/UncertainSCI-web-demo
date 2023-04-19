pyinstaller ^
  --hidden-import vtkmodules.all ^
  --collect-data pywebvue ^
  --onefile ^
  --windowed ^
  --icon uncertainsci-web-demo.ico ^
  .\run.py
