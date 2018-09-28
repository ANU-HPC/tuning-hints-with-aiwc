#!env python3

import pandas
from matplotlib import pyplot as plt

mandelbrot = pandas.read_csv('./mandelbrot.csv',header=None)
mandelbrot[63][63] = 255
plt.imsave('mandelbrot.png',mandelbrot)

mandelbrot_vectorized = pandas.read_csv('./mandelbrot-vectorized.csv',header=None)
mandelbrot_vectorized[63][63] = 255
plt.imsave('mandelbrot-vectorized.png',mandelbrot_vectorized)

#from PIL import Image
#img = Image.fromarray(mandelbrot, 'I')
#img.save('my.png')
#img.show()
