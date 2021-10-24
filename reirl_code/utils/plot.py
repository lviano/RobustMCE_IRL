import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import numpy as np

def add_arrow(pi, shape, mode):
    if mode == "single":
        for s, a in enumerate(pi):    #acs optimal actions
            
            if a == 0: ##up
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]), 0, -0.45, head_width=0.05) 
            if a == 1: ##right
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0.45, 0, head_width=0.05)
            if a == 2: ##down
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0, 0.45, head_width=0.05)
            if a == 3: ##left
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  -0.45, 0, head_width=0.05) 
    if mode == "multiple":
        for s, acs in enumerate(pi):
            for a in acs:
                if a == 0: ##up
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]), 0, -0.45, head_width=0.05) 
                if a == 1: ##right
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0.45, 0, head_width=0.05)
                if a == 2: ##down
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0, 0.45, head_width=0.05)
                if a == 3: ##left
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  -0.45, 0, head_width=0.05) 

    if mode == "max_ent":
        for s, p_acs in enumerate(pi):
            for a, p_a in enumerate(p_acs):
                
                if a == 0: ##up
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]), 0, -0.45, head_width=0.05, alpha = p_a) 
                if a == 1: ##right
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0.45, 0, head_width=0.05, alpha = p_a)
                if a == 2: ##down
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0, 0.45, head_width=0.05, alpha = p_a)
                if a == 3: ##left
                    plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  -0.45, 0, head_width=0.05, alpha = p_a) 


def plot_format(solver, policy):
    """ It transform a random policy expressed by a grid of dimension n_states x n_actions in the format accepted 
        accepted by the function plot_value and policy
    """
    pi_to_plot = [np.argwhere(policy[s,:] > 0).flatten().tolist() for s in range(solver.env.n_states)]
    return pi_to_plot

    
def plot_value_and_policy(sol, policy, title, mode = "multiple", show = False):
    
    pol_to_plot = plot_format(sol, policy) if not (mode == "max_ent") else policy
    plt.matshow(sol.v.reshape(sol.env.size,sol.env.size))
    add_arrow(pol_to_plot, [sol.env.size, sol.env.size], mode)
    plt.colorbar()
    if not(title == None):
        title_list = title.split('_')
        plt.title(' '.join(title_list), verticalalignment = 'bottom')

    if show:
        plt.show()
    else:
        plt.savefig('../plot/'+title+'.png')
        plt.savefig('../plot/'+title+'.pdf')


def plot_on_grid(vector, size, title = None, log_color = False, show = False):
    if log_color:
        plt.matshow(vector.reshape(size, size),
           norm=colors.LogNorm(vmin=vector.min(), vmax=vector.max()),
           cmap='PuBu_r')

    else:
        plt.matshow(vector.reshape(size, size), cmap='PuBu_r')
    plt.colorbar()
    if not(title == None):
        title_list = title.split('_')
        plt.title(' '.join(title_list), verticalalignment = 'bottom')   
    if not show:
        plt.savefig('../plot/'+title+'.png')
        plt.savefig('../plot/'+title+'.pdf')
    else:
        plt.show()

def plot_reward(vector, size, title, tdw = False, show = False):
    matplotlib.rcParams.update({'font.size': 22})
    # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
    if tdw:
        col_dict = { 
            -6: "red",
            -2: "mediumturquoise",
            -1: "lightgrey",
             0: "white" 

        }
        
    else:    
        col_dict={  -100:"mediumturquoise",
                -1:"lightgrey",
                0:"white"}

    # We create a colormar from our list of colors
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

    # Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
    if tdw:
        labels = np.array(["-6", "-2","-1","0"])
    else:
        labels = np.array(["-100", "-1","0"])
    len_lab = len(labels)

    # prepare normalizer
    ## Prepare bins for the normalizer
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    print(norm_bins)
    ## Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    # Plot our figure
    #plt.figure()
    fig,ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(vector.reshape(size,size), cmap=cm, norm=norm)
    #ax.tick_params(labelsize=20)
    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cb = fig.colorbar(im, format=fmt, ticks=tickz)
    #cb.ticklabels_params(size = 20)
    if show:
        plt.show()
    else:
        plt.savefig('../plot/'+title+'.png')
        plt.savefig('../plot/'+title+'.pdf')

def plot_log_lines(list_to_plot, list_name, axis_label, folder, title, x_axis = None, show = False):
    n_iter = list_to_plot[0].size
    plt.figure()
    for _, item in enumerate(zip(list_to_plot, list_name)):
        if x_axis is None:
            plt.loglog(np.arange(n_iter), item[0], label = item[1])
        else:
            plt.loglog(x_axis, item[0], label = item[1])

    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig('../plot/log'+title+'.png')
        plt.savefig('../plot/log'+title+'.pdf')

def plot_lines(list_to_plot, list_name, axis_label, folder, title, x_axis = None, show = False):
    plt.style.use('seaborn')
    n_iter = list_to_plot[0].size
    plt.figure()
    for _, item in enumerate(zip(list_to_plot, list_name)):
        if x_axis is None:
            plt.plot(np.arange(n_iter), item[0], label = item[1])
        else:
            plt.plot(x_axis, item[0], label = item[1])
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig('../plot/'+folder+title+'.png')
        plt.savefig('../plot/'+folder+title+'.pdf')

color_list = ["green", "red", "blue", "orange", "purple", "navy", "black", "skyblue", "darksalmon"]

def plot_lines_and_ranges(  list_to_plot, list_sigmas, list_name, axis_label, folder, title, x_axis = None, show = False, legend = True, ylim=None, color_list_custom = None, vertical=None):
    
    plt.style.use('seaborn')
    n_iter = list_to_plot[0].size
    plt.figure(figsize=(6, 6))
    if vertical is not None:
        xv = float(vertical)
        plt.axvline(x = xv, color = "black") 
    if color_list_custom:
        for i, item in enumerate(zip(list_to_plot, list_name, list_sigmas)):
            if x_axis is None:
                plt.plot(np.arange(n_iter), item[0], label = item[1], color = color_list_custom[i], marker = "o", markersize=8)
                plt.fill_between(np.arange(n_iter), item[0]+item[2], item[0]-item[2],facecolor = color_list_custom[i], alpha=0.1)
            else:
                plt.plot(x_axis, item[0], label = item[1], color = color_list_custom[i], marker = "o", markersize=8)
                plt.fill_between(x_axis, item[0]+item[2], item[0]-item[2],facecolor = color_list_custom[i], alpha=0.1)        
    else:
        for i, item in enumerate(zip(list_to_plot, list_name, list_sigmas)):
            if x_axis is None:
                plt.plot(np.arange(n_iter), item[0], label = item[1], color = color_list[i], marker = "o", markersize=8)
                plt.fill_between(np.arange(n_iter), item[0]+item[2], item[0]-item[2],facecolor = color_list[i], alpha=0.1)
            else:
                plt.plot(x_axis, item[0], label = item[1], color = color_list[i], marker = "o", markersize=8)
                plt.fill_between(x_axis, item[0]+item[2], item[0]-item[2],facecolor = color_list[i], alpha=0.1)
        
    plt.xlabel(axis_label[0], fontsize = 26)
    plt.ylabel(axis_label[1], fontsize = 26)
    plt.tick_params(labelsize=26)
    if ylim:
        plt.ylim(ylim)
    if legend:
        plt.legend(fontsize = 20, frameon = 1, loc='upper center', bbox_to_anchor=(0.4, -0.2),
          fancybox=True, shadow=True, ncol=3)
    if show:
        plt.savefig('../plot/' + folder+title+'.png', bbox_inches='tight')
        plt.savefig('../plot/' + folder+title+'.pdf',bbox_inches='tight')
        plt.show()
    else:
        plt.savefig('../plot/' + folder+title+'.png',bbox_inches='tight')
        plt.savefig('../plot/' + folder+title+'.pdf',bbox_inches='tight')

def plot_log_lines_and_ranges(  list_to_plot, list_sigmas, list_name, axis_label, folder, title, x_axis = None, show = False):
    
    plt.style.use('seaborn')
    n_iter = list_to_plot[0].size
    plt.figure(figsize=(6, 6))
    for i, item in enumerate(zip(list_to_plot, list_name, list_sigmas)):
        if x_axis is None:
            plt.loglog(np.arange(n_iter), item[0], label = item[1], color = color_list[i])
            plt.fill_between(np.arange(n_iter), item[0]+item[2], item[0]-item[2],facecolor = color_list[i], alpha=0.1)
        else:
            plt.loglog(x_axis, item[0], label = item[1], color = color_list[i])
            plt.fill_between(x_axis, item[0]+item[2], item[0]-item[2],facecolor = color_list[i], alpha=0.1)
        
    plt.xlabel(axis_label[0], fontsize = 26)
    plt.ylabel(axis_label[1], fontsize = 26)
    plt.tick_params(labelsize=26)
    plt.legend(fontsize = 26, frameon = 1)
    if show:
        plt.savefig('../plot/' + folder+ 'logfillBetween'+title+'.png', bbox_inches='tight')
        plt.savefig('../plot/' + folder+ 'logfillBetween'+title+'.pdf',bbox_inches='tight')
        plt.show()
    else:
        plt.savefig('../plot/' + folder+ 'logfillBetween'+title+'.png',bbox_inches='tight')
        plt.savefig('../plot/' + folder+ 'logfillBetween'+title+'.pdf',bbox_inches='tight')

colour_list_ow = ["blue", "green", "orange", "red", "purple", "mediumturquoise"]

def plot_objectworld(objectWorld, size, title = None, log_color = False, show = False):
    xs = []
    ys = []
    ocs = [] #Outer Colors
    ics = [] #Inner Colors
    colours = colour_list_ow[:objectWorld.n_colours]
    print(colours)
    for key, obj in objectWorld.objects.items():
        ys.append(key[0])
        xs.append(key[1])
        ocs.append(colours[obj.outer_colour])
        ics.append(colours[obj.inner_colour])
    print("X")
    print(xs)
    print("Y")
    print(ys)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(objectWorld.state_r.reshape(size,size), cmap = "gray")
    ax.set_xticks(np.arange(0.5,size+0.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5,size+0.5, 1), minor=True)
    ax.grid(c = "black", which = "minor")
    ax.scatter(np.array(xs), np.array(ys), c = ics, edgecolors=ocs)
    if show:
        plt.show()
    else:
        plt.savefig('../plot/'+title+'.pdf') 
        plt.savefig('../plot/'+title+'.png') 

def return_best_line(list_to_plot, stds):
    to_plot = np.array(list_to_plot)
    indices = np.argmax(to_plot, axis=0)
    best_stds = []
    for j, std in enumerate(np.array(stds).T):
        best_stds.append(std[indices[j]])
    return np.max(to_plot, axis=0), best_stds
    
