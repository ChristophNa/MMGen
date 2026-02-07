import numpy as np
from enum import Enum, auto
from typing import Callable
from numpy import sin, cos, pi

class TPMSType(Enum):
    GYROID = auto()
    SCHWARZ_P = auto()
    DIAMOND = auto()
    LIDINOID = auto()
    SPLIT_P = auto()
    NEOVIUS = auto()

def gyroid_eq(x, y, z, a):
    cox = cos(2.0 * pi * x / a)
    six = sin(2.0 * pi * x / a)
    coy = cos(2.0 * pi * y / a)
    siy = sin(2.0 * pi * y / a)
    coz = cos(2.0 * pi * z / a)
    siz = sin(2.0 * pi * z / a)
    # Double Gyroid from sourceCode.py:
    # ((six**2)*(coy**2) + (siy**2)*(coz**2) + (siz**2)*(cox**2) + (2*six*coy*siy*coz) + (2*six*coy*siz*cox) + (2*cox*siy*siz*coz))
    return (six**2 * coy**2) + (siy**2 * coz**2) + (siz**2 * cox**2) + \
           2*(six*coy*siy*coz + six*coy*siz*cox + cox*siy*siz*coz)

def schwarz_p_eq(x, y, z, a):
    cox = cos(2.0 * pi * x / a)
    coy = cos(2.0 * pi * y / a)
    coz = cos(2.0 * pi * z / a)
    # Double Schwarz P: (cox**2) + (coy**2) + (coz**2) + 2*(cox*coy) + 2*(cox*coz) + 2*(coy*coz)
    return cox**2 + coy**2 + coz**2 + 2*(cox*coy + cox*coz + coy*coz)

def diamond_eq(x, y, z, a):
    six = sin(2.0 * pi * x / a)
    cox = cos(2.0 * pi * x / a)
    siy = sin(2.0 * pi * y / a)
    coy = cos(2.0 * pi * y / a)
    siz = sin(2.0 * pi * z / a)
    coz = cos(2.0 * pi * z / a)
    # Double Diamond:
    term1 = (six**2)*(siy**2)*(siz**2) + (six**2)*(coy**2)*(coz**2) + (cox**2)*(siy**2)*(coz**2) + (cox**2)*(coy**2)*(siz**2)
    term2 = 2*((six**2)*siy*siz*coy*coz) + 2*(six*(siy**2)*siz*cox*coz) + 2*(six*siy*(siz**2)*cox*coy)
    term3 = 2*(six*coy*(coz**2)*cox*siy) + 2*(six*(coy**2)*coz*cox*siz) + 2*((cox**2)*siy*coz*coy*siz)
    return term1 + term2 + term3

def lidinoid_eq(x, y, z, a):
    six = sin(2.0 * pi * x / a)
    siy = sin(2.0 * pi * y / a)
    siz = sin(2.0 * pi * z / a)
    cox = cos(2.0 * pi * x / a)
    coy = cos(2.0 * pi * y / a)
    coz = cos(2.0 * pi * z / a)
    sixF = sin(2.0 * pi * 2.0 * x / a)
    siyF = sin(2.0 * pi * 2.0 * y / a)
    sizF = sin(2.0 * pi * 2.0 * z / a)
    coxF = cos(2.0 * pi * 2.0 * x / a)
    coyF = cos(2.0 * pi * 2.0 * y / a)
    cozF = cos(2.0 * pi * 2.0 * z / a)
    
    A = sixF*coy*siz
    B = siyF*coz*six
    C = sizF*cox*siy
    D = -coxF*coyF
    E = -coyF*cozF
    F = -cozF*coxF
    G = 0.3
    
    terms = [A, B, C, D, E, F, G]
    res = 0.0
    for t1 in terms:
        for t2 in terms:
            res += t1 * t2
    return res

def split_p_eq(x, y, z, a):
    six = sin(2.0 * pi * x / a)
    siy = sin(2.0 * pi * y / a)
    siz = sin(2.0 * pi * z / a)
    cox = cos(2.0 * pi * x / a)
    coy = cos(2.0 * pi * y / a)
    coz = cos(2.0 * pi * z / a)
    sixF = sin(2.0 * pi * 2.0 * x / a)
    siyF = sin(2.0 * pi * 2.0 * y / a)
    sizF = sin(2.0 * pi * 2.0 * z / a)
    coxF = cos(2.0 * pi * 2.0 * x / a)
    coyF = cos(2.0 * pi * 2.0 * y / a)
    cozF = cos(2.0 * pi * 2.0 * z / a)

    A = 1.1*sixF*siz*coy
    B = 1.1*siyF*six*coz
    C = 1.1*sizF*siy*cox
    D = -0.2*coxF*coyF
    E = -0.2*coyF*cozF
    F = -0.2*cozF*coxF
    G = -0.4*coxF
    H = -0.4*coyF
    I = -0.4*cozF

    terms = [A, B, C, D, E, F, G, H, I]
    res = 0.0
    for t1 in terms:
        for t2 in terms:
            res += t1 * t2
    return res

def neovius_eq(x, y, z, a):
    cox = cos(2.0 * pi * x / a)
    coy = cos(2.0 * pi * y / a)
    coz = cos(2.0 * pi * z / a)
    
    A = 3*cox
    B = 3*coy
    C = 3*coz
    D = 4*(cox*coy*coz)
    
    terms = [A, B, C, D]
    res = 0.0
    for t1 in terms:
        for t2 in terms:
            res += t1 * t2
    # In graded_tpms.py it was -A**2... + t0**2. 
    # Let's keep it consistent with the others where we subtract t**2 at the end.
    return res

TPMS_REGISTRY: dict[TPMSType, Callable] = {
    TPMSType.GYROID: gyroid_eq,
    TPMSType.SCHWARZ_P: schwarz_p_eq,
    TPMSType.DIAMOND: diamond_eq,
    TPMSType.LIDINOID: lidinoid_eq,
    TPMSType.SPLIT_P: split_p_eq,
    TPMSType.NEOVIUS: neovius_eq,
}
