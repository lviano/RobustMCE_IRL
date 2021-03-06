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

#TODO: Cut axis to 0.02
def CompareAlphas_ow_loader(dim, env_seed, noiseL, fix_horizon, low_alpha = False, include_iil = False, legend = True, alphas = None, end_title = "ablation"):
    lr = 0.001
    fix_start = False
    alphaE = "1.0"
    noisesE = ["0.0", "0.05", "0.1", "0.15", "0.2"]
    noiseL = str(noiseL)
    if(fix_horizon):
        data = load_result("fixOWCompareAlphas_env" + str(env_seed) + "size" + str(dim)+"lr"+str(lr)+noiseL +"alphaE"+alphaE+"fix_start"+str(fix_start)+"run0")
    else:
        data = load_result("YT_OWCompareAlphas_env" + str(env_seed) + "size" + str(dim) + "lr" + str(lr) + noiseL \
                            + "alphaE" + alphaE + "fix_start" + str(fix_start))
    Vs_to_plot = data["Vs_to_plot"]
    sigma_Vs_to_plot = data["sigma_Vs"]
    plot_label = data["labels"]
    for j,item in enumerate(zip(Vs_to_plot,sigma_Vs_to_plot)):
        Vs_to_plot[j] = item[0][:-2]
        sigma_Vs_to_plot[j] = item[1][:-2]
    for i,enoise in enumerate(noisesE):
        noisesE[i] = float(enoise)
        
    if not alphas == None:
        current_pos = 0
        for i, alpha in enumerate(["1.0", "0.95", "0.9", "0.85", "0.8"]):
            if alpha not in alphas:
                Vs_to_plot.pop(current_pos) 
                plot_label.pop(current_pos)
                sigma_Vs_to_plot.pop(current_pos)
            else:
                current_pos += 1
    """if not low_alpha:
        Vs_to_plot.pop(5)
        plot_label.pop(5)
        sigma_Vs_to_plot.pop(5)"""
        
    if end_title == "presentation":
        plot_label = ["MCE", "Robust MCE : " + str(alphas[-1]), "expert"]
        #plot_label = ["MCE", "Robust MCE", "expert"]
        
    if include_iil:
        lr_iil = 0.01
        data = load_result( "IIL_OW_CompareAlphas_env" + str(env_seed) + "size" + str(dim)+"lr"+str(lr_iil)+noiseL+"fix_start"+str(fix_start))
        Vs_to_plot.append(data["Vs_to_plot"][0][:-2])
        sigma_Vs_to_plot.append(data["sigma_Vs"][0][:-2])
        plot_label.append(data["labels"][0])
    #Vs_to_plot.insert(0, Vs_to_plot.pop(-1))
    #sigma_Vs_to_plot.insert(0, sigma_Vs_to_plot.pop(-1))
    #plot_label.insert(0, plot_label.pop(-1))
    plot_lines_and_ranges(  list_to_plot = Vs_to_plot,
                            list_sigmas = sigma_Vs_to_plot,
                            list_name = plot_label,
                            axis_label = ["Expert Noise", "Total Return "],
                            folder = "",
                            title = "NotebookCompareAlphasEnvOWnoiseL"+ noiseL+"dim"+str(dim)+"alphaE"+alphaE+"fix_start"+str(fix_start)+"legend"+str(legend)+end_title+str(include_iil),
                            x_axis = noisesE,
                            show = True,
                            legend = legend,
                            vertical = noiseL )
