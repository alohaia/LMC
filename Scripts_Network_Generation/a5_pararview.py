#!/usr/bin/env python

from paraview.simple import *

data = OpenDataFile("./nw_output/graph_gen_process/test_1_semi_realistic_network_final.vtp")
tube = Tube(Input=data)

tube.Radius = 0.1
tube.VaryRadius = data
tube.NumberofSides = 12
Show(tube)
ColorBy(GetDisplayProperties(tube), ('CELLS', 'diameter'))
