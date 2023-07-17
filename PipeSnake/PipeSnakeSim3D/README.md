# pipesnake_visualization_3d

This recreates the old pipe snake visualization to have 3D functionality. It utilizes Python's matplotlib module for plotting. Currently a work in progress.

pipe.py
- Defines the 3D pipe

bot.py
- Defines the pipe snake robot

driver.py
- Contains the main() method to create the pipe
- Run this to see the visualization

config.yaml
- Holds parameters to define parameters of the simulation
- NOT YET IMPLEMENTED

stl_proocessing.py
- renders stl files using open3d
- stl files will either by used for reference in creating the visualization, or will replace matplotlib's plots
