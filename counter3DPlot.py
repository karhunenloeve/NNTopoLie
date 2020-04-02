from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import numpy as np
from numpy import genfromtxt
from colormap import rgb2hex, rgb2hls, hls2rgb
from operator import add

def hex_to_rgb(hex):
     hex = hex.lstrip('#')
     hlen = len(hex)
     return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

def adjust_color_lightness(r, g, b, factor):
    h, l, s = rgb2hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls2rgb(h, l, s)
    return rgb2hex(int(r * 255), int(g * 255), int(b * 255))

def darken_color(r, g, b, factor=0.1):
    return adjust_color_lightness(r, g, b, 1 - factor)

def mix_colors(c1,c2,factor1=0.1,factor2=0.9):
	c1 = map(lambda x: x/255 * factor1, list(c1))
	c2 = map(lambda x: x/255. * factor2, list(c2))
	color = list(map(add, c1, c2))
	return color

def get_data(dimension: float, folder: str):
	files = os.listdir("results/" + folder + "/")

	for i in range(0, len(files)):
		my_data = genfromtxt("results/" + folder + "/" + files[i], delimiter=',').transpose()
		color_bad = hex_to_rgb("#1a1a1d")
		color_good = hex_to_rgb("#6f2232")
		color_perfect = hex_to_rgb("#c3073f")
		
		epochs = my_data[0][1:]
		training_loss = my_data[1][1:]
		validation_loss = my_data[2][1:]
		dim = dimension

		if i <= dim:
			shad_color = mix_colors(mix_colors(color_bad, color_good, 1 - (i / dim), i / dim), (255,255,255), 0.2, 0.8)
			plt.plot(epochs, validation_loss, color=shad_color)
		else: 
			shad_color = mix_colors(color_perfect, color_good, i / len(files), 1 - i / len(files))
			plt.plot(epochs, validation_loss, color=shad_color)

	plt.ylim(ymax = 1, ymin = 0)
	plt.show()