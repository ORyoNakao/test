import dft_ccas
import csv
import numpy as np
import pandas as pd
import time
from readmat import readmat

input_name="_in_font"
#input_name="in_general"

weight="_w_font"
#weight="w_general"

fm="./fm/"

layer_num=1
#kind_of_input 0 -> font
#              1 ->ImageNet
kind_of_input=1


if kind_of_input == 0:
    input_dir="input_font(70)/"
else:
    input_dir="input_ImageNet/"

all_cca=[]

f_layer=1
g_layer=1

f_layer_name="fm_layer"+str(f_layer)
f_path=fm+input_dir+"font/"+f_layer_name+".mat"
conv_act1=readmat(f_path,False)

g_layer_name="fm_layer"+str(g_layer)
g_path=fm+input_dir+"font/"+g_layer_name+".mat"
conv_act2=readmat(g_path,False)

result=dft_ccas.fourier_ccas(conv_act1,conv_act2)
#print("mean:",np.mean(result['mean2'][0]))

mean=(np.mean(result["mean"][0]),np.mean(result["mean"][1]))
print("mean:",mean)

result.to_csv("./test/cca.csv")


        