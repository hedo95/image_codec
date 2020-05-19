import  os
import re
import h5py

def checkPathName(pathSen):
    
    path = os.path.dirname(pathSen)

    try:
        if path: os.makedirs(path, exist_ok = True)
        # Crear el directorio
    except:
        raise Exception(f'No se puede crear el directorio {path}')
        # Indicamos nosotros que path ha fallado, porque el propio Python no lo dice.