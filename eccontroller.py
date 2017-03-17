client = None
lview = None
sendAr = None

import ecworker

def mapComputations(ecComputeList, printLevel=0):
    if len(ecComputeList) == 0:
        return []
    
    if not client:
        resultCounter = 0
        ecTList = []
        for TInd, iElement in ecComputeList:
            if printLevel >= 2:
                print str(TInd) + ' : ' + str(resultCounter) + ' / ' + str(len(ecComputeList))
            ecT = ecworker.computeElementCorrector(iElement)
            ecTList.append(ecT)
            resultCounter += 1
        if printLevel >= 2:
            print 'Done'
        return ecTList
    else:
        sendAr.wait()
        computeElementCorrector = lambda ecCompute: ecworker.computeElementCorrector(ecCompute[1])
        ar = lview.map(computeElementCorrector, ecComputeList)
        ar.wait_interactive()
        return ar.get()

def setupWorker(world, coefficient, IPatchGenerator, k, clearFineQuantities):
    if not client:
        ecworker.setupWorker(world, coefficient, IPatchGenerator, k, clearFineQuantities)
    else:
        global sendAr
        setupWorkerWrapper = lambda x: ecworker.setupWorker(*x)
        sendAr = client[:].apply_async(setupWorkerWrapper, (world, coefficient, None, k, clearFineQuantities))
        
def setupClient(clientIn):
    global client, lview
    client = clientIn
    lview = client.load_balanced_view()
