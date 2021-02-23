import numpy as np
import matplotlib.pyplot as plt

'''
explorng colors. Good demo
https://matplotlib.org/3.1.1/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py
'''

#colors_b = plt.cm.Blues(np.linspace(0.15, 0.5, 3))
#colors_g = plt.cm.Greens(np.linspace(0.15, 0.5, 3))
#N = 5
#menMeans = (20, 35, 30, 35, 27)
#womenMeans = (35, 45, 40, 45, 37)
#menStd = (3, 5, 2, 3, 3)
#capsize = 5.0
#womenStd = (3, 5, 2, 3, 3)
#ind = np.arange(N)    # the x locations for the groups
#width = 0.25       # the width of the bars: can also be len(x) sequence
#
#p1 = plt.bar(ind, menMeans, width, yerr=menStd, color = colors_b[1], alpha = 0.0, capsize = capsize)
#p2 = plt.bar(ind, womenMeans, width,yerr=womenStd, color = colors_b[2], alpha = 0.0, capsize = capsize)
#p3 = plt.bar(ind+width, menMeans, width, yerr=menStd, color = colors_g[1], alpha = 0.0, capsize = capsize)
#p4 = plt.bar(ind+width, womenMeans, width,yerr=womenStd, color = colors_g[2], alpha = 0.0, capsize = capsize)
##
##plt.ylabel('Scores')
##plt.title('Scores by group and gender')
#plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
##plt.yticks(np.arange(0, 81, 10))
##plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Men', 'Women', "m1", "w2"))
#
##plt.show()
#
##plt.setp(markers, marker='D', markersize=10, markeredgecolor="orange", markeredgewidth=2)
#
## create data
#import numpy as np
#values=np.random.uniform(size=40)
# 
##lolipop https://python-graph-gallery.com/181-custom-lollipop-plot/
#
## change color and shape and size and edges
#markers, stemlines, baseline = plt.stem(ind, menMeans)
#plt.setp(markers, markersize=10)
#markers, stemlines, baseline = plt.stem(womenMeans)
#plt.setp(markers, markersize=15, alpha = 0.75)
#
#markers, stemlines, baseline = plt.stem(ind + width, menMeans)
#plt.setp(markers, marker='D', markersize=10, alpha = 0.5) #markeredgecolor="orange", markeredgewidth=2,
#markers, stemlines, baseline = plt.stem(ind + width, womenMeans)
#plt.setp(markers, marker='D', markersize=15, alpha = 0.5)
#plt.show()



def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    #bs_prop_list =[0.05, 0.10, 0.25]
    labels = [0.05, 0.10, 0.25]
    i = 0
    for rect in ax.patches:
       # print(i)
#        if i == 3:
#            i = 0
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)
        #label = "BS: 0.0.05, BS: 0.10"
        
        label = str(i)
        print(x_value, i)
        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.
        i += 1
     #   break