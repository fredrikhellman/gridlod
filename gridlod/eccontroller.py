client = None
lview = None
sendAr = None

from . import ecworker
import ipyparallel as ipp

def mapComputations(ecComputeList, printLevel=0):
    if len(ecComputeList) == 0:
        return []
    
    if not client:
        resultCounter = 0
        ecTList = []
        for TInd, iElement in ecComputeList:
            if printLevel >= 2:
                print(str(TInd) + ' : ' + str(resultCounter) + ' / ' + str(len(ecComputeList)))
            ecT = ecworker.computeElementCorrector(iElement)
            ecTList.append(ecT)
            resultCounter += 1
        if printLevel >= 2:
            print('Done')
        return ecTList
    else:
        sendAr.wait_interactive()
        computeElementCorrector = lambda ecCompute: ecworker.computeElementCorrector(ecCompute[1])
        ar = lview.map(computeElementCorrector, ecComputeList)
        ar.wait_interactive()
        return ar.get()

def setupWorker(world, coefficient, IPatchGenerator, k, clearFineQuantities, printLevel):
    if not client:
        ecworker.setupWorker(world, coefficient, IPatchGenerator, k, clearFineQuantities)
    else:
        global sendAr
        setupWorkerWrapper = lambda x: ecworker.setupWorker(*x)
        if hasattr(coefficient, 'rCoarse'):
            ar = client[:].apply_async(setupWorkerWrapper, (world, None, None, k, clearFineQuantities))
            ar.wait()
            ar = client[:].apply_async(ecworker.hasaBase)
            hasaBase = ar.get()
            if any(h is False for h in hasaBase):
                if printLevel >= 2:
                    print('Sending large coefficient')
                sendAr = client[:].apply_async(ecworker.sendar, coefficient._aBase, coefficient._rCoarse)
            else:
                if printLevel >= 2:
                    print('Sending small coefficient')
                sendAr = client[:].apply_async(ecworker.sendar, None, coefficient._rCoarse)
        elif hasattr(coefficient, 'aLagging'):
            ar = client[:].apply_async(setupWorkerWrapper, (world, None, None, k, clearFineQuantities))
            ar.wait()
            if printLevel >= 2:
                print('Sending both coefficients')
            sendAr = client[:].apply_async(ecworker.sendas, coefficient._aFine, coefficient._aLagging)
        else:
            sendAr = client[:].apply_async(setupWorkerWrapper, (world, coefficient, None, k, clearFineQuantities))

def clearWorkers():
    if not client:
        return
    else:
        ar = client[:].apply_async(ecworker.clearWorker)
        ar.wait()
        
def setupClient(clientIn):
    global client, lview
    client = clientIn
    lview = client.load_balanced_view()
    clearWorkers()
