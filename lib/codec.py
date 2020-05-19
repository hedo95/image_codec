import numpy as np
import h5py
from lib.models.jpeg import Codec
from lib.utils.utils import *

def coder(imagen, tamBlq, nomFic, nomCodec='Codec',
               **optsCodec):

    if isinstance(tamBlq, int): tamBlq = tamBlq, tamBlq
    tamBlq = tamBlq + imagen.shape[2:]

    numFil, numCol = imagen.shape[:2]
    numBlq = int(numFil / tamBlq[0]), \
             int(numCol / tamBlq[1])
    numFil, numCol = int(numBlq[0] * tamBlq[0]), \
                     int(numBlq[1] * tamBlq[1])

    imagen = imagen[:numFil, :numCol]

    codec = eval(f'{nomCodec}(imagen, tamBlq, nomCodec,'
                 f'**optsCodec)')

    imgCod = np.ndarray(numBlq + codec.tamCod,
                        dtype=codec.tipCof.dtype)

    for fil in range(0, numFil, tamBlq[0]):
        for col in range(0, numCol, tamBlq[1]):
            bloque = imagen[fil: fil + tamBlq[0],
                            col: col + tamBlq[1]]
            imgCod[fil // tamBlq[0], col // tamBlq[1]] = \
                codec.codifica(bloque)

    checkPathName(nomFic)
    with h5py.File(nomFic, 'w', rdcc_nbytes=4) as h5pySal:
        codec.meteInfo(h5pySal)
        h5pySal.create_dataset('imgCod', data=imgCod)
