# you must include '%matplotlib notebook' for this to work
import matplotlib.pyplot as plt
#%matplotlib notebook

# generate plot function
#def plt_dynamic(x, y1, y2, y3, y4, color_y1='g', color_y2='b', color_y3='y', color_y4='r'):
def plt_dynamic(x, y1, color_y1='g'):
    sub1.plot(x, y1, color_y1)
    #sub1.plot(x, y2, color_y2)
    #sub1.plot(x, y3, color_y3)
    #sub2.plot(x, y4, color_y4)
    fig.canvas.draw()

time_limit = 5
y1_lower = 0
y1_upper = 400

#y2_lower = -5
#y2_upper = 5

# create plots
fig, sub1 = plt.subplots(1,1)
sub2 = sub1.twinx()

# set plot boundaries
sub1.set_xlim(0, time_limit)  # this is typically time
sub1.set_ylim(y1_lower, y1_upper)  # limits to your y1
#sub2.set_xlim(0, time_limit) # time, again
#sub2.set_ylim(y2_lower, y2_upper) # limits to your y2

# set labels and colors for the axes
sub1.set_xlabel('time (s)', color='k')
sub1.set_ylabel('x[green], y [blue], z[yellow]')
sub1.tick_params(axis='x', colors='k')
sub1.tick_params(axis='y', colors="g")

#sub2.set_ylabel('rewawrds', color='r')
#sub2.tick_params(axis='y', colors='r')
