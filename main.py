from lib.codec import *
from lib.decodec import *
import matplotlib.pyplot as plt

path = ''
filename = ''
ext = '.png'
file = path + filename + ext
blocksize = 400
im = plt.imread(file)

coder(im,blocksize,'images/result.hdf5')

im = decoder('images/result.hdf5')
plt.imshow(im)
plt.show()