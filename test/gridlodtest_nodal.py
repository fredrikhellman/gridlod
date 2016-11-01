import gridlod
import numpy as np

from pyevtk.hl import imageToVTK 
import scipy.sparse as sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

import sys

NWorldCoarse = np.array([10, 10, 10])
NCoarseElement = np.array([10, 10, 10])
NWorldFine = NWorldCoarse*NCoarseElement

NpFine = np.prod(NWorldFine+1)

g = gridlod.grid.grid(NWorldCoarse, NCoarseElement)
gp = gridlod.grid.gridPatch(g)

p = gp.pFine()

a = np.ones(NWorldFine, order='F')
apFlat = a.flatten(order='F')

MLoc = gridlod.fem.localMassMatrix(gp.NPatchFine)
MFull = gridlod.fem.assemblePatchMatrix(gp.NPatchFine, MLoc)

ALoc = gridlod.fem.localStiffnessMatrix(gp.NPatchFine)
AFull = gridlod.fem.assemblePatchMatrix(gp.NPatchFine, ALoc, apFlat)

fixedMask = np.abs(np.prod(np.abs(p-0.5)-0.5, axis=1)) == 0
freeMask  = np.logical_not(fixedMask)

#INodal = gridlod.interp.nodalPatchMatrix(gp.iPatchCoarse, gp.NPatchCoarse, g.NWorldCoarse, g.NCoarseElement)

print "aha"
sys.stdout.flush()
exit()
print "oho"

up = np.sin(2*np.pi*p[:,0])*np.sin(2*np.pi*p[:,1])*np.sin(2*np.pi*p[:,2])
fp = 3*4*(np.pi**2)*up

bFull = MFull*fp
bFree = bFull[freeMask]

AFree = AFull[freeMask][:,freeMask]
uFineFree,_ = sparse.linalg.cg(AFree, bFree)

uFine = np.zeros(NpFine)
uFine[freeMask] = uFineFree

uFineCube = uFine.reshape(NWorldFine+1)

print np.sqrt(np.dot(uFine-up, AFull*(uFine-up)))

imageToVTK("./image", pointData = {"u" : uFineCube} )
