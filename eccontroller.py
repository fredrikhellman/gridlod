client = None
lview = None

import ecworker

class ParallelResult:
    def __init__(self, ar, iElement):
        self.ar = ar
        self.iElement = iElement
        
    def get(self):
        #ecT = ecworker.computeElementCorrector(self.iElement)
        #        self.ar.wait()
        #        return ecT
        return self.ar.get()[0]
    
class LazyResult:
    def __init__(self, iElement):
        self.iElement = iElement
        
    def get(self):
        ecT = ecworker.computeElementCorrector(self.iElement)
        return ecT

def enqueue(iElement):
    if not client:
        return LazyResult(iElement)
    else:
        computeElementCorrector = lambda x: ecworker.computeElementCorrector(x)
        ar = lview.map(computeElementCorrector, [iElement])
        ar.wait()
        return ParallelResult(ar, iElement)
    
def setupWorker(world, coefficient, IPatchGenerator, k, clearFineQuantities):
    if not client:
        ecworker.setupWorker(world, coefficient, IPatchGenerator, k, clearFineQuantities)
    else:
        setupWorkerWrapper = lambda x: ecworker.setupWorker(*x)
        ar = client[:].apply_async(setupWorkerWrapper, (world, coefficient, None, k, clearFineQuantities))
        ar.wait()
        
def setupClient(clientIn):
    global client, lview
    client = clientIn
    lview = client.load_balanced_view()
