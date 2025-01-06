import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

def my_plot(plotting_func):
    """This is the decorator for plotting with the desired rc parameters. 

        It saves the resulting figure in the path: Exercises Lesson 8

         It takes as input a plotting function having the following input parameters:
        - 'x' is the array of the x axis
         - 'y' is the array of the y axis
        - 'x_label' is a string for the label of the x axis
        - 'y_label' is a string for the label of the y axis
        - 'title' is the title of the plot"""
    
    def wrapper(*args):
        # set the rc params for plotting
        with mpl.rc_context({"font.family" : "serif", 
                              "mathtext.fontset" : "cm",
                              "font.size" : 11,
                              "figure.figsize" : [6,6],
                              "lines.linewidth" : 1,
                              "axes.prop_cycle" : cycler(color=['black','red','blue','orange','green']),
                            }):
            
            # build the figure object
            fig = plt.figure()
    
            # create the axis for the plot
            ax = fig.add_subplot(111)
    
            ax.set_xlabel(args[2])
            ax.set_ylabel(args[3])
            ax.set_title(args[4])
        
            # call the function
            plotting_func(*args)
    
        # save the plot
        plt.savefig(r"c:\Users\Ludovico\Desktop\unimib\PhD\Scientific Computing\scientificcomputing_bicocca_2024\working\Exercises Lesson 8\%s.pdf"%args[4])

        # show the image 
        plt.show()
        
    return wrapper
    
    
@my_plot
def plot_funct(x,y,x_label,y_label,title):
    """This function returns a matplotlib plot object.
       It requires 5 parameters as input:
       - 'x' is the array of the x axis
       - 'y' is the array of the y axis
       - 'x_label' is a string for the label of the x axis
       - 'y_label' is a string for the label of the y axis
       - 'title' is the title of the plot"""
    
    return plt.plot(x,y)
