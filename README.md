# UncertainSCI-web-demo
This is a demonstration of UncertainSCI deployed on the web


Built with [trame-cookiecutter](https://github.com/Kitware/trame-cookiecutter)

To build the trame app:
```
cd uncertainsci-web-demo
pip install . # Install your new application

pip install pywebview  # For app usage
pip install jupyterlab # For Jupyter usage
```

to run locally:
```
uncertainsci-web-demo
```
As a desktop app:
```
uncertainsci-web-demo --app
```
as a notebook
```
jupyter-lab
```
then load the generated `examples/jupyter/show.ipynb` file
