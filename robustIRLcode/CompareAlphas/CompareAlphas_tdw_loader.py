import pickle
import sys
sys.path.insert(0,'../src/')

from optimizers import *
from environment import *
from IRLalgorithms import *
from MDPsolver import *
from utils import *
from plot import *
import argparse
import copy


def CompareAlphas_tdw_loader(dim, env_type, noiseL, low_alpha = False, alphas = None, end_title= "ablation", legend = True):
    lr = 0.5
    fix_start = False
    alphaE = "1.0"
    noisesE = ["0.0", "0.05", "0.1", "0.15", "0.2"]
    noiseL = str(noiseL)

    data = load_result( "CompareAlphas_tdw_env" + str(env_type) + "size" + str(dim)+"lr"+str(lr)+noiseL+"alphaE"+alphaE+"fix_start"+str(fix_start))
    Vs_to_plot = data["Vs_to_plot"]
    sigma_Vs_to_plot = data["sigma_Vs"]
    plot_label = data["labels"]
    ###


    for i,enoise in enumerate(noisesE):
        noisesE[i] = float(enoise)
    if not low_alpha:
        if noiseL == "0.0":
            Vs_to_plot.pop(6) 
            plot_label.pop(6)
            sigma_Vs_to_plot.pop(6)
            Vs_to_plot.pop(5) 
            plot_label.pop(5)
            sigma_Vs_to_plot.pop(5)
            
    if not alphas == None:
        current_pos = 0
        for i, alpha in enumerate(["1.0", "0.95", "0.9", "0.85", "0.8"]):
            if alpha not in alphas:
                Vs_to_plot.pop(current_pos) 
                plot_label.pop(current_pos)
                sigma_Vs_to_plot.pop(current_pos)
            else:
                current_pos += 1
    if end_title == "presentation":
        plot_label = ["MCE", "Robust MCE : " + str(alphas[-1]), "expert"]
    
    plot_lines_and_ranges(  list_to_plot = Vs_to_plot,
                            list_sigmas = sigma_Vs_to_plot,
                            list_name = plot_label,
                            axis_label = ["Expert Noise", "Total Return "],
                            folder = "",
                            title = "NotebookCompare Alphas_tdw Env "+str(env_type)+" noise L "+ noiseL+"dim"+str(dim)+"alphaE"+alphaE+"fix_start"+str(fix_start)+end_title,
                            x_axis = noisesE,
                            show = True,
                            legend = legend,
                            vertical = noiseL)
