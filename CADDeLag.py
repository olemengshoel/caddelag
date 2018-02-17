# SAMPLE CALL:
# distributed_cad_v1.py
# distributed_cad_v1.py 50K dense $SPARK_HOME $NOBACKUP_BASE
# -----------------------------------------------------------
# distributed_cad_v1.py GRAPH_NODES SPARSE_GRAPH SPARK_HOME NOBACKUP_BASE
# GRAPH_NODES can have K at the end to indicate 1e3
# SPARSE_GRAPH should be sparse for sparse computations
# SPARK_HOME must be set properly. If it is not passed, 
#      the default parameter must be set correctly. 
#      
# -------------------------------------------------------------
import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg.distributed import BlockMatrix, RowMatrix, DistributedMatrix, CoordinateMatrix, IndexedRowMatrix, IndexedRow
from pyspark.mllib.linalg import DenseVector, Vectors
from pyspark.mllib.linalg import Matrices, SparseMatrix, DenseMatrix, Matrix
from pyspark.rdd import RDD
from pyspark.sql import SQLContext
import numpy as np
from scipy.io import loadmat
from scipy import sparse
import scipy.io as sio
from datetime import datetime
import os, re
import gc
import mmap
import math
import logging
import time
import itertools

import construct_graphs as constructGraphs
# --------------------------------------
GRAPH_NODES = 10
SPARSE_GRAPH = False
SPARK_HOME = 'BASEPATH/Spark/spark-2.1.0/'
NOBACKUP_SCRATCH_BASE = 'BASEPATH/USERNAME/SCRATCH/'

MIN_BLOCK_SIZE = 4000
# --------------------------------------
def findAnomalies(zfile1, zfile2):
    logging.warn('In findAnomalies')
    numTopAnomalousEdges = 20
    numAnomalousNodes, numContributors = 100, 10
    n, p = GRAPH_NODES, SQUARE_BLOCK_SIZE
    global RESULTS_DIR, RESULTS_dict

    anomaliesMatFile = RESULTS_DIR + 'anomalies-' + str(n) + '.mat'
    matricesDir = NOBACKUP_SCRATCH_BASE + str(GRAPH_NODES) + '/matrices/'
    if not os.path.exists(RESULTS_DIR):
        os.system('mkdir -p ' + RESULTS_DIR)

    if not blocksExistInFiles(matricesDir + 'A1'):
        A1 = constructGraphs.createAdjMat(n, 12, SPARSE_GRAPH, p, sc)
        writeBlocksToFile(A1, 'A1', matricesDir)
    if not blocksExistInFiles(matricesDir + 'A2'):
        A2 = constructGraphs.createAdjMat(n, 16, SPARSE_GRAPH, p, sc)
        writeBlocksToFile(A2, 'A2', matricesDir)
    if not blocksExistInFiles(matricesDir + 'D1'):
        D1 = ctdFromEmbedding(zfile1)
        writeBlocksToFile(D1, 'D1', matricesDir)
    if not blocksExistInFiles(matricesDir + 'D2'):
        D2 = ctdFromEmbedding(zfile2)
        writeBlocksToFile(D2, 'D2', matricesDir)

    if not blocksExistInFiles(matricesDir + 'deltaE'):
        deltaE = constructDeltaE(matricesDir)
        writeBlocksToFile(deltaE, 'deltaE', matricesDir)
    else:
        deltaE = loadBlocksFromFiles(matricesDir + 'deltaE')

    def topKInBlock(block):
        I, J = block[0]
        if I < J:
            return []
        vec = block[1].toArray().flatten()
        ids = vec.argsort()[-numTopAnomalousEdges:]
        topEdges = []
        for edgeId in ids:
            i, j = edgeId / p, edgeId % p
            topEdges.append((vec[edgeId], (I*p + i, J*p + j)))
        return topEdges
        
    topAnomalyEdgesAllBlocks = deltaE.flatMap(topKInBlock).collect()
    logging.warn('Done colelct 2')
    topAnomalyEdges = sorted(topAnomalyEdgesAllBlocks, key = lambda x: x[0])[-numTopAnomalousEdges:][::-1]
    constructGraphs.electionEdgesReport(topAnomalyEdges)

    nodeAnomalyScores = matrixVectorMultiply(deltaE, np.ones(n))
    topAnomalousNodes = nodeAnomalyScores.argsort()[::-1][\
        :min(numAnomalousNodes, n)]
    print 'Top anomalous nodes: ', topAnomalousNodes

    nodeIdContributorsList = findTopContributors(deltaE, \
        topAnomalousNodes, numContributors)

    def getDeltaERow(rowId):
        rowBlockId = rowId / p
        row = np.zeros(n)
        for j in range(n/p):
            colBlock = loadBlockAsNpMat(matricesDir + 'deltaE_' + \
                        str(rowBlockId) + '_' + str(j) + '.mat')
            row[(j*p) : ((j+1) * p)] = colBlock[rowId - (rowBlockId * p),:]
        return row
    topRows = sc.parallelize(topAnomalousNodes, numAnomalousNodes).map(getDeltaERow).collect()
    sio.savemat(anomaliesMatFile, {'nodeAnomalyScores':nodeAnomalyScores,\
                                   'nodeIdContributorsList': nodeIdContributorsList, \
                                   'topAnomalyRows': topRows})
    print 'nodeIdContributorsList \n', nodeIdContributorsList
    constructGraphs.generateReport(nodeIdContributorsList)

    return

def findTopContributors(deltaE, topAnomalousNodes, numContributors):
    def topKinRows(block):
        mat = block[1].toArray()
        n = mat.shape[0]
        I, J = block[0][0] * n, block[0][1] * n
        idTopKs = []
        for i in topAnomalousNodes:
            if I <= i < I + n:
                topKindices = mat[i-I,:].argsort()[::-1][:numContributors]
                topKs = [(mat[i-I, j], J+j) for j in topKindices]
                idTopKs.append((i, topKs))
        return idTopKs
    def extractTopK(p, q):
        topKs = sorted(p + q, key = lambda x: x[0])[::-1][:numContributors]
        return topKs
    topKBlocks = deltaE.flatMap(topKinRows)
    nodeToContributors = topKBlocks.reduceByKey(extractTopK).collect()
    nodeIdContributorsList = np.zeros((len(topAnomalousNodes), \
        numContributors + 1) , dtype = 'int')
    for i, nodeConts in enumerate(nodeToContributors):
        nodeIdContributorsList[i,0] = nodeConts[0]
        nodeIdContributorsList[i,1:] = [x[1] for x in nodeConts[1]]        
    return nodeIdContributorsList

def constructDeltaE(path):
    logging.warn('  constructDeltaE ')
    n, p = GRAPH_NODES, SQUARE_BLOCK_SIZE
    def f(x):
        A1 = loadBlockAsNpMat(path + 'A1_' + idToStr(x) + '.mat')
        A2 = loadBlockAsNpMat(path + 'A2_' + idToStr(x) + '.mat')
        D1 = loadBlockAsNpMat(path + 'D1_' + idToStr(x) + '.mat')
        D2 = loadBlockAsNpMat(path + 'D2_' + idToStr(x) + '.mat')
        D1, D2 = D1 ** 2, D2 ** 2
        delA, delD = abs(A1 - A2), abs(D1 - D2)
        delE = np.multiply(delA, delD)
        return (x, npToDenseMat(delE))
    return allBlocksIds().map(f)

def constructDelta(path, prefix):
    n, p = GRAPH_NODES, SQUARE_BLOCK_SIZE
    def f(x):
        A1 = loadBlockAsNpMat(path + prefix + '1_' + idToStr(x) + '.mat')
        A2 = loadBlockAsNpMat(path + prefix + '2_' + idToStr(x) + '.mat')
        delA = abs(A1 - A2)
        return (x, npToDenseMat(delA))
    return allBlocksIds().map(f)

def ctdFromEmbedding(zfile):
    Z = loadmat(zfile)['Z']
    n, p = GRAPH_NODES, SQUARE_BLOCK_SIZE
    blocks, numBlocks = constructGraphs.splitInBlocks(Z, p)
    
    # It is crucial to set the number of blocks correctly
    ZblocksRdd = sc.parallelize(blocks, numBlocks) 
    # Also crucial to cache the Z matrix, which is small n x k_RP
    ZblocksRdd = ZblocksRdd.cache()
    n,f = ZblocksRdd.count(), ZblocksRdd.first()
    logging.warn('ZblocksRdd = ' + str(ZblocksRdd.getNumPartitions()))
    ZblockPairs = ZblocksRdd.cartesian(ZblocksRdd)
    D = ZblockPairs.map(eucledianDistances)
    return D

def eucledianDistances(ZblockPair):
    I, J = int(ZblockPair[0][0]), int(ZblockPair[1][0])
    blockI, blockJ = ZblockPair[0][1], ZblockPair[1][1]
    allCombinations = itertools.product(blockI, blockJ)
    allCombsEdges = [np.linalg.norm(p[0] - p[1]) for p in allCombinations]
    n = blockI.shape[0]
    if len(allCombsEdges) == (n*n):
        adj = np.reshape(allCombsEdges, (n,n))
    else:
        adj = np.zeros((n,n))
    G = Matrices.dense(n,n, adj.transpose().flatten())
    return ((I,J), G)

def writeBlocksToFile(blocks, prefix, basePath):
    def writeBlock(block):
        i, j = block[0]
        filename = basePath + prefix + '_' + str(i) + '_' + str(j) + '.mat'
        data = {'G' : block[1].toArray(), 'block_id' : block[0]}
        sio.savemat(filename, data)
        return str(block[0])

    if not os.path.exists(basePath):
        os.system('mkdir -p ' + basePath)
    writeStatusFilename = basePath + prefix + '_write_stat.txt'
    # "Start" indicates start of file writing.
    os.system('echo Start > ' + writeStatusFilename)
    writeStats = blocks.map(writeBlock)
    n = writeStats.count()
    # "End" indicates end of file writing.
    os.system('echo End > ' + writeStatusFilename)
    logging.warn('Done writing ' + prefix + ' to ' + basePath)

def blocksExistInFiles(pathsPrefix):
    writeStatusFilename = pathsPrefix + '_write_stat.txt'
    if not os.path.exists(writeStatusFilename):
        return False
    file = open(writeStatusFilename,'r')
    line = file.read()
    file.close()
    return 'End' in line

def loadBlocksFromFiles(pathsPrefix):
    n, p = GRAPH_NODES, SQUARE_BLOCK_SIZE
    def prepareFileList():
        naiveWay = True
        filelistName = pathsPrefix + 'filelist.txt'
        cmd = 'find ' + pathsPrefix + '*.mat | sort > ' + filelistName
        os.system(cmd)
        return filelistName

    def loadBlockFromMatFile(filename):
        data = loadmat(filename, squeeze_me = True)
        id, G = data['block_id'], data['G']
        if isinstance(G, sparse.csc_matrix):
            sub_matrix = Matrices.sparse(p, p, G.indptr, G.indices, G.data)        
        else:
            sub_matrix = Matrices.dense(p, p, G.transpose().flatten())        
        return ((id[0], id[1]), sub_matrix)

    numBlocks = (n/p) ** 2
    filelistRdd = sc.textFile(prepareFileList(), minPartitions = numBlocks)
    blocks_rdd = filelistRdd.map(loadBlockFromMatFile)
    return blocks_rdd

def npToDenseMat(ndArr):
    m, n = ndArr.shape
    return Matrices.dense(m, n, ndArr.transpose().flatten())

def idToStr(id):
    return str(id[0]) + '_' + str(id[1])

def loadBlockAsNpMat(blockFileName):
    p = SQUARE_BLOCK_SIZE
    if not os.path.exists(blockFileName):
        return np.zeros((p, p))
    data = loadmat(blockFileName)
    return data['G'] 


# --------------------------------------------------------------
def commuteTimeDistancesEmbed(A_block_mat, tolerance, epsilon, d):
    logging.warn('Starting CTD for ' + str(GRAPH_NODES) + ' nodes')
    n, p = GRAPH_NODES, SQUARE_BLOCK_SIZE
    global RESULTS_DIR, RESULTS_dict, EXACT_SOLVE_INPUTS, SCRATCH_COUNTER

    rudeSolverResultPath = NOBACKUP_SCRATCH_BASE + str(GRAPH_NODES) + '/rude-res/'
    if blocksExistInFiles(rudeSolverResultPath + 'ChainProduct2'):
        ChainProduct2 = loadBlocksFromFiles(rudeSolverResultPath + 'ChainProduct2') 
        filename = RESULTS_DIR + 'exact-solve-inputs.mat'
        data = loadmat(filename)
        Chi, q = data['Chi'], int(data['q'])
        SCRATCH_COUNTER = 70
        return exactSolverCMU(ChainProduct2, Chi, q)

    scale, tolProb = min(100, int(math.ceil(math.log(n,2)) / epsilon)), 0.5
    RESULTS_dict['scale'] = scale

    d0_A, d1_A = rowSumD(A_block_mat)
    logging.warn('max d0_A: ' + str(max(d0_A)))
    if blocksExistInFiles(rudeSolverResultPath + 'D1_Cprod'):
        D1_Cprod = loadBlocksFromFiles(rudeSolverResultPath + 'D1_Cprod')
        SCRATCH_COUNTER = 30
    else:
        ChainProduct = rudeSolverCMU(A_block_mat, d1_A, d)
        D1_Cprod = diagonalMultiply(d1_A, ChainProduct)
        logging.warn('Rude solver end') 
        writeBlocksToFile(D1_Cprod, 'D1_Cprod', rudeSolverResultPath)
    D1_Cprod = D1_Cprod.cache()

    if blocksExistInFiles(rudeSolverResultPath + 'ChainProduct2'):
        ChainProduct2 = loadBlocksFromFiles(rudeSolverResultPath + 'ChainProduct2')
        SCRATCH_COUNTER = 40
    else:
        d0_minus_a0 = A_block_mat.map(lambda x : dMinusA(x, d0_A))
        logging.warn('Calling naiveMultiply(D1_Cprod, d0_minus_a0)')
        ChainProduct2 = naiveMultiply(D1_Cprod, d0_minus_a0)   # .cache()
        writeBlocksToFile(ChainProduct2, 'ChainProduct2', rudeSolverResultPath)
        logging.warn('Done ChainProduct2')

    logging.warn('Calling randomProjectWB')
    Y = randomProjectWB(A_block_mat, n, scale, tolProb)
    Chi = matrixVectorMultiply(D1_Cprod, Y)

    q = int(math.ceil(math.log(1.0 / tolerance))) * 1
    # TODO: Change this???
    filename = RESULTS_DIR + 'exact-solve-inputs.mat'
    sio.savemat(filename, {'Chi':Chi, 'q':q})
    return exactSolverCMU(ChainProduct2, Chi, q)

def rudeSolverCMU(A0, d1_A0, d):
    global RESULTS_dict
    n, p = GRAPH_NODES, SQUARE_BLOCK_SIZE
    S0_powers = A0.map(lambda x : normalizeLaplacian(x, d1_A0))
    C = diagonalBlockMatrix(d1_A0)

    for i in range(d-1):
        logging.warn('Rude solve: ' + str(i) + '/' + str(d-2))
        if i > 0:
            S0_powers = naiveMultiply(S0_powers, S0_powers, True)
        S0plusI = S0_powers.map(addOneToDiagonal)
        C = naiveMultiply(S0plusI, C)

    return C

def exactSolverCMU(ChainProduct2, Chi, q):
    global RESULTS_dict
    ChainProduct2 = ChainProduct2.cache()
    Z = Chi  #  First iteration
    for k in range(q-2):
        logging.warn('In q loop:' + str(k) + '/' + str(q-1))
        # Z = Z - naiveMultiply(ChainProduct2, Z) + Chi
        temp = matrixVectorMultiply(ChainProduct2, Z)
        Z = Z - temp + Chi

    RESULTS_dict['q'] = q
    return Z

# ASSUMPTION: Square block size must be <= n
# D0: row_sums 
# D1: row_sums ^ (-0.5)
def rowSumD(block_mat):
    n = GRAPH_NODES
    global RESULTS_DIR, SCRATCH_COUNTER
    dFile = RESULTS_DIR + 'D_' + str(n) + '.mat'
    logging.warn('rowSumD: D found = ' + str(os.path.exists(dFile)))
    if os.path.exists(dFile):
        data = loadmat(dFile)
        d = data['D0']
        D0_array = np.array(d[0])
        # matrixVectorMultiply increments the counter
        SCRATCH_COUNTER += 1
    else:
        if not os.path.exists(RESULTS_DIR):
            os.system('mkdir -p ' + RESULTS_DIR)
        D0_array = matrixVectorMultiply(block_mat, np.ones(n))
        sio.savemat(dFile, {'D0':D0_array})
    
    zerosInD = D0_array == 0
    D1_array = np.nan_to_num(D0_array ** (-0.5))
    D1_array[D1_array == np.inf] = 0 
    D1_array[zerosInD] = 0
    return D0_array, D1_array

# If the result already exists in the temp directory, 
# just read from file and return. Otherwise, do the 
# computation and save to file for future use.
def randomProjectWB(A_block_mat, n, scale, tolProb):
    global NOBACKUP_SCRATCH
    nobackup = NOBACKUP_SCRATCH + 'qwb/'
    if not os.path.exists(nobackup):
        os.system('mkdir -p ' + nobackup)
    n, p = GRAPH_NODES, SQUARE_BLOCK_SIZE

    def randomSignBuildColumns(block):        
        I, J = block[0]
        if I < J:                                                                                                      
            return str(block[0])                                                                                                  
        # np.random.seed(I + J)
        if isinstance(block[1], SparseMatrix):
            # TODO: FIX THIS
            pass
        else:
            mat = np.copy(block[1].toArray())
            # Only use the upper triangular matrix for W * B
            if I == J:
                mat = np.triu(mat)      
            
            mat[mat > 0] = mat[mat > 0] ** 0.5
            matSize = mat.shape[0]
            rowsQwb, colsQwb = np.zeros((p, scale)), np.zeros((p, scale))
            for t in range(scale):
                ons = np.random.choice([-1, 1], \
                      size=(matSize, matSize), p=[tolProb, 1-tolProb])
                randMat = mat * ons
                rowSums, colSums = np.sum(randMat, axis = 1), np.sum(randMat, axis = 0)
                # print 'WB[', I, J, '] ', t, ': ', rowSums, colSums
                rowsQwb[:, t] = rowSums
                colsQwb[:, t] = -1.0 * colSums
            rowsFile = nobackup + str(I) + '_source_' + str(I) + '_' + str(J) + '.mat'
            if I == J:
                totalQwb = rowsQwb + colsQwb
                sio.savemat(rowsFile , {'wb' : totalQwb})
            else:
                colsFile = nobackup + str(J) + '_source_' + str(I) + '_' + str(J) + '.mat'
                sio.savemat(rowsFile , {'wb':rowsQwb})
                sio.savemat(colsFile, {'wb':colsQwb})
        return str(block[0])

    def onsTimesWB():
        status = A_block_mat.map(randomSignBuildColumns).collect()
        rowIdToFiles = mapRowIDToAllFiles(nobackup)
        numBlocks = n / p
        rowBlockIds = sc.parallelize(range(numBlocks), numBlocks)
        yRowIdsBlocks = rowBlockIds.map(lambda i :addBlocksByRowId(i, \
                                    rowIdToFiles[i], nobackup, 'wb')).collect()
        Y = np.zeros((n, scale))
        for i, mat in yRowIdsBlocks:
            Y[i*p:(i+1)*p, :] = mat
        return Y

    global RESULTS_DIR
    yFile = RESULTS_DIR + 'Y_' + str(n) + '.mat'
    if os.path.exists(yFile):
        logging.warn('Found Y in file')
        data = loadmat(yFile)
        Y = data['Y']
    else:
        Y = onsTimesWB()
        logging.warn('Done onsTimesWB')
        Y = Y / (scale ** 0.5)
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        sio.savemat(yFile, {'Y':Y})
    return Y

def addToResults(** nameBlockMatrices):
    global RESULTS_dict
    n, p = GRAPH_NODES, SQUARE_BLOCK_SIZE
    for name, blockMat in nameBlockMatrices.iteritems():
        RESULTS_dict[name] = BlockMatrix(blockMat, p, p, n, n).toLocalMatrix().toArray()

def mapRowIDToAllFiles(baseDir):
    allfiles = os.listdir(baseDir)
    idFiles = {}
    for file in allfiles:
        if file[-4:] == '.mat':
            id = int(file.split('_')[0])
            if not id in idFiles:
                idFiles[id] = []
            idFiles[id].append(file)
    return idFiles
                
def addBlocksByRowId(I, allfiles, baseDir, varName):
    sumMat = loadmat(baseDir + allfiles[0])[varName]
    for file in allfiles[1:]:
        sumMat += loadmat(baseDir + file)[varName]
    return (I, sumMat)

def sumRowBlocksAsMat(baseDir, varName):
    rowIdToFiles = mapRowIDToAllFiles(baseDir)
    n, p = GRAPH_NODES, SQUARE_BLOCK_SIZE
    numBlocks = n / p
    rowBlockIds = sc.parallelize(range(numBlocks), numBlocks)
    yRowIdsBlocks = rowBlockIds.map(lambda i :addBlocksByRowId(i, \
                            rowIdToFiles[i], baseDir, varName)).collect()
    nCols = yRowIdsBlocks[0][1].shape[1]
    Y = np.zeros((n, nCols))
    for i, mat in yRowIdsBlocks:
        Y[i*p:(i+1)*p, :] = mat
    return Y
# --------------------------------------------------------------
# Multiplication using nobackup
# C = A * B
# A: blocks
# B: blocks
def naiveMultiply(A, B, squareA = False):
    if not isinstance(A, RDD):
        return diagonalMultiply(A, B, diagLeft = True)
    if not isinstance(B, RDD):
        return diagonalMultiply(B, A, diagLeft = False)        
    logging.warn('Starting naiveMultiply')
    global NOBACKUP_SCRATCH, SCRATCH_COUNTER
    blocksPath = NOBACKUP_SCRATCH + str(SCRATCH_COUNTER) + '/'    
    if not os.path.exists(blocksPath):
        os.makedirs(blocksPath)    

    if not blocksExistInFiles(blocksPath + 'A'):
        logging.warn('Not found ' + blocksPath + 'A')
        writeBlocksToFile(A, 'A', blocksPath)
    if not squareA and not blocksExistInFiles(blocksPath + 'B'):
        logging.warn('Not found ' + blocksPath + 'B')
        writeBlocksToFile(B, 'B', blocksPath)
    def productBlock(id):
        N, p = GRAPH_NODES, SQUARE_BLOCK_SIZE
        num_blocks = N/p
        prod = np.zeros((p,p))    
        for j in range(num_blocks):
            left = loadBlockAsNpMat(blocksPath + \
                'A_' + str(id[0]) + '_' + str(j) + '.mat') 
            if squareA:
                right = loadBlockAsNpMat(blocksPath + \
                    'A_' + str(j) + '_' + str(id[1]) + '.mat')
            else:
                right = loadBlockAsNpMat(blocksPath + \
                    'B_' + str(j) + '_' + str(id[1]) + '.mat')
            if left is not None and right is not None:
                prod += np.dot(left, right)
        return (id, npToDenseMat(prod))

    C = allBlocksIds().map(productBlock)
    SCRATCH_COUNTER += 1
    logging.warn('Done naiveMultiply')
    return C

def diagonalMultiply(diag, A, diagLeft = True):
    def f(block):
        I, J = block[0]
        mat, p = block[1].toArray(), SQUARE_BLOCK_SIZE
        prod = np.zeros((p, p))
        if diagLeft:
            for i in range(p):
                prod[i,:] = mat[i,:] * diag[I*p + i]
        else:
            for j in range(p):
                prod[:,j] = mat[:,j] * diag[J*p + j]
        return (block[0], npToDenseMat(prod))
    return A.map(f)

def normalizeLaplacian(block, d1):
    I, J = block[0]
    mat, p = block[1].toArray(), SQUARE_BLOCK_SIZE
    L = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            L[i, j] = mat[i, j] * d1[I*p + i] * d1[J*p + j]
    nomalizedL = Matrices.dense(p, p, L.transpose().flatten())
    return (block[0], nomalizedL)

def matrixVectorMultiply(matBlocks, vec):
    return MatrixVectorMultiplyReduce(matBlocks, vec)

def MatrixVectorMultiplyReduce(mat, vec):
    global NOBACKUP_SCRATCH, SCRATCH_COUNTER
    nobackup = NOBACKUP_SCRATCH + str(SCRATCH_COUNTER) + '/'
    SCRATCH_COUNTER += 1
    if not os.path.exists(nobackup):
        os.makedirs(nobackup)    
    
    vecIs1D = False
    if vec.ndim == 1:
        vec = np.reshape(vec, (len(vec), 1))
        vecIs1D = True
    def MultiplyBlockVec(block):
        rowId, colId, x = block[0][0], block[0][1], block[1]
        if isinstance(x, SparseMatrix):
            b = sparse.csc_matrix((x.values, x.rowIndices, x.colPtrs), shape=(x.numRows, x.numCols))
        else:       
            b = x.toArray()
        blockSize = SQUARE_BLOCK_SIZE  
        if len(vec.shape) > 1:
            subVector = vec[(colId*blockSize):(colId*blockSize + blockSize), :]
        else:
            subVector = vec[(colId*blockSize):(colId*blockSize + blockSize)]
        prod = b.dot(subVector)
        I, J = rowId, colId
        prodFile = nobackup + str(I) + '_source_' + str(I) + '_' + str(J) + '.mat'
        sio.savemat(prodFile, {'prod': prod})
        return str(block[0])
           
    logging.warn('MatrixVectorMultiplyReduce: Writing to ' + nobackup)
    status = mat.map(MultiplyBlockVec).collect()
    productMat = sumRowBlocksAsMat(nobackup, 'prod')
    if vecIs1D:
        return productMat.flatten()
    return productMat
# --------------------------------------------------------------
def diagonalBlockMatrix(diag):
    n, p = len(diag), SQUARE_BLOCK_SIZE
    def difun(x, vect):
        if x[0] == x[1]:
            sm = SparseMatrix(p, p, np.linspace(0, p, num = (p+1)), \
                np.linspace(0, p-1, num = p), vect[(x[0]*p):((x[0]+1)*p)])
            return (x, sm)
        else:
            h = sparse.csc_matrix((p,p))
            return (x, Matrices.sparse(p,p,h.indptr,h.indices,h.data))
    num_blocks = int(n / p)
    if SPARSE_GRAPH:
        blockids = sc.parallelize(itertools.product(xrange(num_blocks),repeat=2))
        blocksRdd = blockids.map(lambda x: difun(x,diag))
    else:
        blocks = []
        for i in range(num_blocks):
            blocks.append(difun((i, i), diag))
        blocksRdd = sc.parallelize(blocks, num_blocks)        
    D = blocksRdd
    return D

def dMinusA(block, d_array):
    I, J = block[0]
    mat, p = block[1].toArray(), SQUARE_BLOCK_SIZE
    if I == J:
        d = d_array[(I*p) : ((I + 1)*p)]
        mat = np.diag(d) - mat
    else:
        mat = -1 * mat 
    return (block[0], npToDenseMat(mat))

def addOneToDiagonal(block):
    I, J = block[0]
    if not I == J:
        return block
    mat, p = block[1].toArray(), SQUARE_BLOCK_SIZE
    return (block[0], npToDenseMat(mat + np.eye(p)))

def allBlocksIds():
    N = GRAPH_NODES
    p = SQUARE_BLOCK_SIZE
    num_blocks = N/p
    i = range(num_blocks)
    allKeys = [x for x in itertools.product(i, i)]
    return sc.parallelize(allKeys, num_blocks * num_blocks)

def cleanupInterimDirs():
    global EXACT_SOLVE_INPUTS, NOBACKUP_SCRATCH
    rudeSolverResultPath = NOBACKUP_SCRATCH_BASE + str(GRAPH_NODES) + '/rude-res/'
    dFile = RESULTS_DIR + 'D_' + str(n) + '.mat'
    yFile = RESULTS_DIR + 'Y_' + str(n) + '.mat'
    os.system('rm -rf ' + NOBACKUP_SCRATCH + '/*')
    os.system('rm -rf ' + EXACT_SOLVE_INPUTS + '/*')
    os.system('rm -rf ' + rudeSolverResultPath + '/*')
    os.system('rm -rf ' + dFile)
    os.system('rm -rf ' + yFile)

if __name__=='__main__':
    if len(sys.argv) > 1:
        size = sys.argv[1]
        if size[-1].upper() == 'K':
            GRAPH_NODES = int(size[:-1])*1000
        else:
            GRAPH_NODES = int(size)
        if GRAPH_NODES >= 3e5:
            MIN_BLOCK_SIZE = 10000
        if GRAPH_NODES == 256e3:
            MIN_BLOCK_SIZE = 4000 * 1
    if len(sys.argv) > 2 and sys.argv[2].lower() == 'sparse':           
        SPARSE_GRAPH = True
    if len(sys.argv) > 3:           
        SPARK_HOME = sys.argv[3]
        if not SPARK_HOME[-1] == '/':
            SPARK_HOME += '/'
    if len(sys.argv) > 4:           
        NOBACKUP_SCRATCH_BASE = sys.argv[4]
        if not NOBACKUP_SCRATCH_BASE[-1] == '/':
            NOBACKUP_SCRATCH_BASE += '/'
    global RESULTS_DIR, RESULTS_dict, EXACT_SOLVE_INPUTS, NOBACKUP_SCRATCH
    RESULTS_DIR = NOBACKUP_SCRATCH_BASE + str(GRAPH_NODES) + '/results/'            
    EXACT_SOLVE_INPUTS = NOBACKUP_SCRATCH_BASE + str(GRAPH_NODES) + '/exact-in/'    
    NOBACKUP_SCRATCH = NOBACKUP_SCRATCH_BASE + str(GRAPH_NODES) + '/scratch/'

    if not os.path.exists(NOBACKUP_SCRATCH):
        os.system('mkdir -p ' + NOBACKUP_SCRATCH)
        os.system('lfs setstripe -c 1 ' + NOBACKUP_SCRATCH)
    
    if GRAPH_NODES < MIN_BLOCK_SIZE:
        SQUARE_BLOCK_SIZE = GRAPH_NODES / 2
        minP = 1
    else:
        SQUARE_BLOCK_SIZE = MIN_BLOCK_SIZE # 2000
        minP = pow(GRAPH_NODES/SQUARE_BLOCK_SIZE, 2)    

    global SCRATCH_COUNTER
    SCRATCH_COUNTER = 0

    tol = 1e-4  
    epsilon = 1e-3 
    d = 3 
    # np.random.seed(0) 
    # np.random.seed(1337) # To match with MATLAB
    # ---------------------------LOGGING----------------------------------------------                                  
    logfname = SPARK_HOME + 'log_size_' + str(GRAPH_NODES) + '_' + \
        datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.log'
    logging.basicConfig(filename=logfname, filemode='w', level= logging.INFO, \
        format='%(asctime)s:%(levelname)s:%(message)s', \
        datefmt='%m/%d/%Y %I:%M:%S %p')      
    logging.warn(sys.argv[0] + '\n SPARK_HOME = ' + SPARK_HOME \
                 + '\n p = ' + str(SQUARE_BLOCK_SIZE))   
    # ----------------Create new Spark config---------------------------------------------       
    # 0 means unlimited; if driver fails, set some value like 16g
    conf = SparkConf().set("spark.driver.maxResultSize", "32g")
    conf.set( "spark.akka.frameSize", "2040")
    sc = SparkContext(conf=conf, appName="Commute time distances ")
    sqlContext = SQLContext(sc)
    sc.addFile(SPARK_HOME + "construct_graphs.py")

    # ----------------------------------------------------------------------------------    
    n, p = GRAPH_NODES, SQUARE_BLOCK_SIZE
    zfile1, zfile2 = RESULTS_DIR + 'elections-12-'+ str(n) + '-Z.mat', \
                 RESULTS_DIR + 'elections-16-'+ str(n) + '-Z.mat'        

    if not os.path.exists(zfile1):
        RESULTS_dict = {}
        A1 = constructGraphs.createAdjMat(n, 12, SPARSE_GRAPH, p, sc)
        Z1 = commuteTimeDistancesEmbed(A1, tol, epsilon, d)
        RESULTS_dict['Z'] = Z1
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        sio.savemat(zfile1, RESULTS_dict)

        cleanupInterimDirs()
    SCRATCH_COUNTER = 101

    if not os.path.exists(zfile2):    
        logging.warn('doing next CTD')
        RESULTS_dict = {}
        A2 = constructGraphs.createAdjMat(n, 16, SPARSE_GRAPH, p, sc)
        Z2 = commuteTimeDistancesEmbed(A2, tol, epsilon, d)
        RESULTS_dict['Z'] = Z2    
        sio.savemat(zfile2, RESULTS_dict)
    
    SCRATCH_COUNTER = 201

    RESULTS_dict = {}
    findAnomalies(zfile1, zfile2)
    logging.warn('Completed Run!')






    
