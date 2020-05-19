import numpy as np
import h5py
from lib.models.jpeg import *

def decoder(file):
    with h5py.File(file, 'r') as h5pyEnt:
            nomCodec = h5pyEnt['codec'][()]
            codec = eval(f'{nomCodec}(h5pyEnt=h5pyEnt)')
            imgCod = h5pyEnt['imgCod'][...]

    imagen = np.ndarray(codec.shape, dtype=codec.pixel.dtype)
    numFil, numCol = codec.shape[:2]

    for fil in range(0, numFil, codec.tamBlq[0]):
        for col in range(0, numCol, codec.tamBlq[1]):
            blqCod = imgCod[fil // codec.tamBlq[0],
                            col // codec.tamBlq[1]]
            imagen[fil: fil + codec.tamBlq[0],
                col: col + codec.tamBlq[1]] = \
                codec.decodifica(blqCod)

    return imagen