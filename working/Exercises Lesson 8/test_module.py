import module_myPlot
import module_myPlot.my_plot_module
import numpy as np

x = np.linspace(0,2*np.pi)
y = np.cos(x)
x_label = 'x'
y_label = 'cos(x)'
title = 'myplot cosx'

module_myPlot.my_plot_module.plot_funct(x,y,x_label,y_label,title)

