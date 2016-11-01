import gridlod
import numpy as np

from pyevtk.hl import imageToVTK 
import scipy.sparse as sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

NWorld = np.array([20, 20, 20])
Np = np.prod(NWorld+1)

g = gridlod.grid.grid(NWorld)
gp = gridlod.grid.gridPatch(g)

p = gp.pCoarse()

a = np.ones(NWorld, order='F')
apFlat = a.flatten(order='F')

MLoc = gridlod.fem.localMassMatrix(gp.NPatchCoarse)
MFull = gridlod.fem.assemblePatchMatrix(gp.NPatchCoarse, MLoc)

ALoc = gridlod.fem.localStiffnessMatrix(gp.NPatchCoarse)
AFull = gridlod.fem.assemblePatchMatrix(gp.NPatchCoarse, ALoc, apFlat)

fixedMask = np.abs(np.prod(np.abs(p-0.5)-0.5, axis=1)) == 0
freeMask  = np.logical_not(fixedMask)

up = np.sin(2*np.pi*p[:,0])*np.sin(2*np.pi*p[:,1])*np.sin(2*np.pi*p[:,2])
fp = 3*4*(np.pi**2)*up

bFull = MFull*fp
bFree = bFull[freeMask]

AFree = AFull[freeMask][:,freeMask]
uFree,_ = sparse.linalg.cg(AFree, bFree)

u = np.zeros(Np)
u[freeMask] = uFree

uCube = u.reshape(NWorld+1)

print np.sqrt(np.dot(u-up, AFull*(u-up)))

imageToVTK("./image", pointData = {"u" : uCube} )
