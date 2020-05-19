import numpy as np
import h5py

def PSNR(ori, cod):
    mse = np.sum((ori - cod) ** 2) / cod.size
    return 20 * np.log10(1) - 10 * np.log10(mse)

class Codec:
    def __init__(self, imagen=None, tamBlq=None, nomCodec=None,
                 *, h5pyEnt=None, **optsCodec):
        if h5pyEnt:
            self.cogeInfo(h5pyEnt)
            return

        self.shape = imagen.shape
        self.pixel = np.array([], dtype=imagen.dtype)
        self.tamBlq = tamBlq
        self.nomCodec = nomCodec
        self.optsCodec = optsCodec
        for clave in optsCodec:
            self.__setattr__(clave, optsCodec[clave])

        # Por defecto, el tamaño y tipo del bloque codificado
        # es igual al tamaño y tipo del bloque original
        self.tamCod = tamBlq
        self.tipCof = np.array([], dtype=imagen.dtype)

    def meteInfo(self, h5pySal):
        dt = h5py.special_dtype(vlen=str)
        h5pySal.create_dataset('codec',
                               dtype=dt, data=self.nomCodec)

        h5pySal.create_dataset('shape', data=self.shape)
        h5pySal.create_dataset('pixel', data=self.pixel)
        h5pySal.create_dataset('tamBlq', data=self.tamBlq)

        h5pySal.create_dataset('tamCod', data=self.tamCod)
        h5pySal.create_dataset('tipCof', data=self.tipCof)

        optsCodec = h5pySal.create_group('optsCodec')
        for clave in self.optsCodec:
            optsCodec.create_dataset(f'{clave}', data=clave,
                                     dtype=dt)
            h5pySal.create_dataset(clave,
                                   data=self.__getattribute__(clave))

    def cogeInfo(self, h5pyEnt):
        self.nomCodec = h5pyEnt['codec'][()]

        self.shape = h5pyEnt['shape'][...]
        self.pixel = h5pyEnt['pixel'][...]
        self.tamBlq = h5pyEnt['tamBlq'][...]

        self.tamCod = h5pyEnt['tamCod'][...]
        self.tipCof = h5pyEnt['tipCof'][...]

        for clave in h5pyEnt['optsCodec']:
            self.__setattr__(clave, h5pyEnt[clave][...])

    def codifica(self, bloque):
        return bloque

    def decodifica(self, blqCod):
        return blqCod


class UInt8(Codec):
    def __init__(self, imagen=None, tamBlq=None, nomCodec=None,
                 *, h5pyEnt=None, **optsCodec):
        super().__init__(imagen, tamBlq, nomCodec,
                         h5pyEnt=h5pyEnt, *optsCodec)

        if not h5pyEnt:
            self.tamCod = tamBlq
            self.tipCof = np.array([], dtype='uint8')

    def codifica(self, bloque):
        return np.uint8(bloque * 255)

    def decodifica(self, blqCod):
        return blqCod / 255

from scipy.fftpack import dct, idct

dct2 = lambda im: dct(dct(im, axis=0), axis=1) / 4 / im.size
idct2 = lambda Im: idct(idct(Im, axis=0), axis=1)

class JPCec(Codec):
    def __init__(self, imagen=None, tamBlq=None, nomCodec=None,
                 *, h5pyEnt=None, **optsCodec):
        super().__init__(imagen, tamBlq, nomCodec,
                         h5pyEnt=h5pyEnt, **optsCodec)

        if not h5pyEnt:
            self.totCof = self.numCof ** 2
            if self.totCof % 2: self.totCof += 1
            self.tamCod = (1 + self.totCof // 2,)
            self.tipCof = np.array([], dtype='uint8')

    def codifica(self, bloque):
        numCof = self.numCof

        if len(bloque.shape) not in (2, 3):
            raise Exception('El bloque ha de ser 2D o 3D')
        elif len(bloque.shape) is 3:
            bloque = bloque[..., 0]

        C = np.array(dct2(bloque)[: numCof, : numCof])
        C.resize(self.totCof)

        Cmax = np.max(np.abs(C[1:])) if numCof > 1 else 1
        lNorm = np.ceil(2 * np.log2(Cmax))
        lNorm = max(min(lNorm, 0), -15)
        norm = 2 ** (lNorm / 2)

        numBytes = 1 + C.size // 2
        blqCod = np.ndarray(numBytes, dtype='uint8')

        blqCod[0] = np.uint8(C[0] * 255)
        blqCod[1] = np.uint8(lNorm + 15) << 4

        cofCod = np.uint8(8 + np.round(8 * C / norm))
        cofCod[np.where(cofCod > 15)] = 15

        blqCod[2:] = cofCod[2:: 2] << 4
        blqCod[1:] |= cofCod[1:: 2]

        return blqCod
    
    def cuantifica(self,blqCod,K):
    
        blqQ = np.zeros(8,8)
        Q = np.array([[16,11,10,16,24,40,51,61],
                      [12,12,14,19,26,58,60,55],
                      [14,13,16,24,40,57,69,56],
                      [14,17,22,29,51,87,80,62],
                      [18,22,37,56,68,109,103,77],
                      [24,35,55,64,81,104,113,92],
                      [49,64,78,87,103,121,120,101],
                      [72,92,95,98,112,100,103,99]])
                      
        for row in range(8):
            for col in range(8):
                blqQ[row][col] = blqCod[row][col] / (K*Q[row][col])

        return blqQ
 

    def descuantifica(self,blqQ,K):
    
        blqCod = np.zeros(8,8)
        Q = np.array([[16,11,10,16,24,40,51,61],
                      [12,12,14,19,26,58,60,55],
                      [14,13,16,24,40,57,69,56],
                      [14,17,22,29,51,87,80,62],
                      [18,22,37,56,68,109,103,77],
                      [24,35,55,64,81,104,113,92],
                      [49,64,78,87,103,121,120,101],
                      [72,92,95,98,112,100,103,99]])
            
        for row in range(8):
            for col in range(8):
                blqCod[row][col] = blqQ[row][col] * K *Q[row][col]

        return blqCod
     

    def decodifica(self, blqCod):
        numCof = self.numCof

        cofCod = np.ndarray(2 * ((numCof ** 2 + 1) // 2))

        cofCod[0] = blqCod[0] / 255
        lNorm = (blqCod[1] >> 4) - 15
        norm = 2 ** (lNorm / 2)
        cofCod[1:: 2] = np.array((blqCod[1:] & 0xF) - 8,
                                 dtype='int8')
        cofCod[2:: 2] = np.array((blqCod[2:] >> 4) - 8,
                                 dtype='int8')

        cofCod[1:] *= norm / 8

        cofCod.resize((numCof, numCof))

        cofCod = cofCod.reshape(numCof, numCof)
        trnCos = np.zeros((self.tamBlq[0], self.tamBlq[1]))
        trnCos[: numCof, : numCof] = cofCod

        blqDec = idct2(trnCos)

        if len(self.tamBlq) is 2:
            bloque = blqDec
        elif self.tamBlq[2] < 4:
            blqDec = blqDec[..., np.newaxis]
            bloque = blqDec
            for chn in range(2, self.tamBlq[2] - 1):
                bloque = np.stack((bloque, blqDec), axis=2)
        elif self.tamBlq[2] == 4:
            bloque = np.stack((blqDec, blqDec, blqDec, np.ones(blqDec.shape)), axis=2)
        else:
            raise Exception(f'No se reconoce el formato {self.tamBlq}')

        return bloque