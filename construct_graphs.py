import itertools
import logging
import math
import numpy as np
import os
import pandas
from pyspark import SparkContext
from pyspark.mllib.linalg import Matrices, SparseMatrix, DenseMatrix
from pyspark.mllib.linalg.distributed import BlockMatrix
from pyspark.sql import SQLContext
from scipy import sparse
from scipy.io import loadmat
import scipy.io as sio

from pyspark.context import SparkContext

GENERATE_SPARSE = False

election_data_path = 'data/'
elecion_files = ['common-ids.csv', 'part1.csv', 'part2.csv']

def edgeDefinitionElection(vx, vy):
    # vx = vx / max(1.0, 1.0 * max(abs(vx)))
    # vy = vy / max(1.0, 1.0 * max(abs(vy)))
    # return np.sum(np.minimum(vx, vy))
    similarity = 0
    for i in range(len(vx)):
        x, y = vx[i], vy[i]
        if (x <= 1e3 and y <= 1e3) or \
           ((1e3 < x <= 1e4) and (1e3 < y <= 1e4)) or \
           (x > 1e4 and y > 1e4):
            similarity += math.log(max(1, min(x, y)))
    return similarity
    
def createAdjMat(graphNodes, year, sparseG, blockSize, sc, data = 'election'):
    if data == 'toy' or graphNodes == 18:
        return createAdjMatToy(graphNodes, year, sparseG, blockSize, sc)
    if graphNodes == 70e3:
        return createSmallAdjMatClimate(graphNodes, year, sparseG, blockSize, sc)
    if graphNodes == 256e3:
        return createAdjMatClimate(graphNodes, year, sparseG, blockSize, sc)
    if data == 'election' or graphNodes > 10e3:
        return createAdjMatElection(graphNodes, year, sparseG, blockSize, sc)
    logging.warn('createAdjMat: Invalid input [data]')
    return None

def createAdjMatToy(graphNodes, year, sparseG, blockSize, sc):
    index = (year > 12) + 1 # 12 maps to 1, 16 maps to 2
    path = 'BASE_PATH/toy_example/'
    A = loadmat(path + 'toy_A' + str(index) + '.mat')['G']
    n = A.shape[0]
    p = n / 2
    subMatrices = [A[:p,:p], A[:p,p:], A[p:,:p], A[p:,p:]]
    ids, blocks = [(0,0), (0,1), (1,0), (1,1)], []
    for i, id in enumerate(ids):
        adj = subMatrices[i]
        G = Matrices.dense(p,p, adj.transpose().flatten())
        blocks.append((id, G))
    blocksRdd = sc.parallelize(blocks, len(ids))
    return blocksRdd    

def createAdjMatClimate(graphNodes, year, sparseG, blockSize, sc):
    # 1994 Jan vs 1995 Jan
    id = (year > 12) + 4  # 12 maps to 4, 16 maps to 5
    dataFile = 'BASE_PATH/climate_precip/preDataVectors.mat'  # precipitation data
    dataVec = loadmat(dataFile, squeeze_me = True)['pre199' + str(id) + '01']
    numBlocks = int(graphNodes / blockSize)
    dataBlocks = [(i/blockSize, dataVec[i:i+blockSize]) for i in range(0, graphNodes, blockSize)]
    dataBlocksRdd = sc.parallelize(dataBlocks, numBlocks)
    allPairBlocks = dataBlocksRdd.cartesian(dataBlocksRdd)

    def radialBasisBlock(pairData):
        I, J = int(pairData[0][0]), int(pairData[1][0])
        dataI, dataJ = pairData[0][1], pairData[1][1]
        n = len(dataI)
        allCombinations = itertools.product(dataI, dataJ)
        allCombsEdges = [radialBasisKernel(p[0], p[1]) for p in allCombinations]
        print 'allCombsEdges ', len(allCombsEdges), (n*n)
        if len(allCombsEdges) == (n*n):
            adj = np.reshape(allCombsEdges, (n,n))
        else:
            adj = np.zeros((n,n))
        if I==J:
            adj[range(n), range(n)] = 0
        G = Matrices.dense(n,n, adj.transpose().flatten())
        return ((I,J), G)

    adjMatBlocks = allPairBlocks.map(radialBasisBlock).cache()
    return adjMatBlocks

def radialBasisKernel(x, y, sigma = 388.0):
    delta = (x - y) ** 2
    return math.exp(-delta/(2.0 * sigma ** 2))

def createSmallAdjMatClimate(graphNodes, year, sparseG, block_size, sc):
    index = (year > 12) + 13  # 12 maps to 13, 16 maps to 14  
    prefix = 'BASE_PATH/climate_precip/GJan'
    data_file = prefix + str(index) + 'padded.mat'
    adj_mat = loadmat(data_file)['G']
    
    n = adj_mat.shape[0]
    if not n == graphNodes:
        logging.warn('Error in size')
    num_blocks, blocks = n / block_size, []

    for i in range(num_blocks):
        start_row, end_row = i * block_size, (i+1) * block_size
        for j in range(num_blocks):
            start_col, end_col = j * block_size, (j+1) * block_size
            if end_row > n or end_col > n:
                print '\n\n\n Error: createSmallAdjMatClimate \n\n\n'
            G = adj_mat[start_row:end_row, start_col:end_col]
            blocks.append(((i,j), G))

    def denseBlock(x):
        adj = x[1].toarray()
        G_dm = Matrices.dense(block_size, block_size, adj.transpose().flatten())
        return (x[0], G_dm)
    # blocksRdd = allBlocksIds(graphNodes, blockSize, sc).map(denseBlock)
    blocksRdd = sc.parallelize(blocks, num_blocks **2).map(denseBlock)
    return blocksRdd    

def allBlocksIds(N, p, sc):
    num_blocks = N/p
    i = range(num_blocks)
    allKeys = [x for x in itertools.product(i, i)]
    return sc.parallelize(allKeys, num_blocks ** 2)


def createAdjMatElection(graphNodes, year, sparseG, blockSize, sc):
    if sparseG:
        GENERATE_SPARSE = True
    normalizeDonations = False
    path = election_data_path
    if (year == 12):
        donations = loadContributionsCSV(path, elecion_files[1])
    elif (year == 16):
        donations = loadContributionsCSV(path, elecion_files[2])
        # Fix to overflow errors due to large Aij 
        donations[4,:] = 0.1 * donations[4,:]
        donations[6,:] = 0.1 * donations[6,:]
    else:
        print 'ERROR: Wrong value of year (', year, '), only 12 and 16 allowed'        
        return

    if graphNodes < len(donations):
        if False:
            d1 = loadContributionsCSV(path, elecion_files[1])
            d2 = loadContributionsCSV(path, elecion_files[2])
            totalD = np.sum(d1 + d2, 1)
            topDonors = totalD.argsort()[-graphNodes:]
            donations = donations[topDonors,:]
        else:
            donations = donations[0:graphNodes,:]

    if graphNodes > len(donations):
        extraNodes = graphNodes - len(donations)
        donations = np.append(donations, np.zeros((extraNodes, donations.shape[1])), axis = 0)

    blocks, n = splitInBlocks(donations, blockSize)
    donationsRdd = sc.parallelize(blocks, n)

    #--------------------------------------    
    n = donationsRdd.count()
    logging.warn('donationsRdd count = ' + str(n) + ', parts = ' + \
        str(donationsRdd.getNumPartitions()))
    donationsRdd.repartition(n).cache()
    a = donationsRdd.take(1)
    sqlContext = SQLContext(sc)
    #--------------------------------------

    logging.warn('donationsRdd parts = ' + str(donationsRdd.getNumPartitions()))

    allPairDonations = donationsRdd.cartesian(donationsRdd)
    logging.warn('allPairDonations count = ' + str(allPairDonations.count()))
    adjMatBlocks = allPairDonations.map(constructElectionBlock)
    if normalizeDonations:
        logging.warn('Before normalizeDonations')
        adjMatBlocks = nomalizeBlocks(adjMatBlocks)
        logging.warn('Done normalizeDonations')
    return adjMatBlocks

    logging.warn('Calling BlockMatrix(), size = ' + str(N))
    adjMat = BlockMatrix(adjMatBlocks, blockSize, blockSize, N, N)    
    return adjMat

def constructElectionBlock(pairDonations):
    I = int(pairDonations[0][0])
    J = int(pairDonations[1][0])
    donationsI = pairDonations[0][1]
    donationsJ = pairDonations[1][1]

    n = donationsI.shape[0]
    allCombinations = itertools.product(donationsI, donationsJ)
    allCombsEdges = [edgeDefinitionElection(p[0], p[1]) for p in allCombinations]
    if len(allCombsEdges) == (n*n):
        adj = np.reshape(allCombsEdges, (n,n))
    else:
        adj = np.zeros((n,n))
    if I==J:
        adj[range(n), range(n)] = 0

    if GENERATE_SPARSE:
        G = sparse.csc_matrix(adj)
        subMatrixSparse = Matrices.sparse(n, n, G.indptr, G.indices, G.data)
        return ((I,J), subMatrixSparse)
    else:
        G = Matrices.dense(n,n, adj.transpose().flatten())
        return ((I,J), G)

def splitInBlocks(donations, blockSize):    
    N, d = donations.shape
    numBlocks = int(math.ceil(1.0 * N / blockSize))
    delta = numBlocks * blockSize - N
    donations = np.vstack((donations, np.zeros((delta, d))))

    N = donations.shape[0]
    m = N/numBlocks
    listOfDonationBlocks = [(i/m, donations[i:i+m,:]) for i in range(0, N, m)]    
    return listOfDonationBlocks, numBlocks

def createElectionDiffGraph(blockSize, sc):
    A1 = createAdjMat(12, blockSize, sc)
    A2 = createAdjMat(16, blockSize, sc)
    logging.warn('\n A1: ' + str(A1.numRows()) + ' x ' + str(A1.numCols()) + ' \n A2: ' + str(A2.numRows()) + ' x ' + str(A2.numCols()))
    logging.warn('\n A1: ' + str(A1.rowsPerBlock) + ' x ' + str(A1.colsPerBlock) + ' \n A2: ' + str(A2.rowsPerBlock) + ' x ' + str(A2.colsPerBlock))
    
    diffA = subtractSparseBlockMatFast(A1, A2, blockSize)
    return diffA

def subtractSparseBlockMatFast(left, right, p):
    n = right.numRows()
    negBlocks = right.blocks.map(lambda x: elementWiseMultiply(x, -1))
    negative_right = BlockMatrix(negBlocks, p, p, n, n)

    result = negative_right.add(left)
    return result

def elementWiseMultiply(block, M):
    mat = block[1]
    mat.values = 1.0*M*(mat.values)
    return (block[0], mat)

def WriteEdgesPerBlock(b, SQUARE_BLOCK_SIZE):
    I = b[0][0]
    J = b[0][1]
    if I > J:
        return [] # ((I,J), 0)
    if isinstance(b[1], SparseMatrix):
        mat = sparse.csc_matrix((b[1].values, b[1].rowIndices, b[1].colPtrs), shape=(b[1].numRows, b[1].numCols))
        links = sparse.find(mat)
        edges = zip((links[0]+ (I * SQUARE_BLOCK_SIZE)), (links[1]+ (J * SQUARE_BLOCK_SIZE)),links[2])
    else:
        mat = np.array(b[1].toArray())
        if I == J:
            mat = np.triu(mat)
        i,j = np.nonzero(mat)
        values = mat[i,j]
        i = i + I * SQUARE_BLOCK_SIZE
        j = j + J * SQUARE_BLOCK_SIZE
        edges = []		
        for ind in range(len(values)):
            edges.append((i[ind], j[ind], values[ind]))	
    return edges

def nomalizeBlocks(adjMatBlocks):
    blockMaxs = adjMatBlocks.map(blockMaxWithCheck)
    matrixMax = blockMaxs.reduce(max)
    normalizedBlocks = adjMatBlocks.map(lambda x: elementWiseDivide(x, matrixMax))
    return normalizedBlocks

def blockMaxWithCheck(block):
    if len(block[1].values) == 0:
        return 0.0
    else:
        return max(block[1].values)

# does block/M
def elementWiseDivide(block, M):
    mat = block[1]
    mat.values = 1.0*(mat.values)/M
    return (block[0], mat)


def loadContributionsCSV(path, filename):
    names = pandas.read_csv(path + elecion_files[0], sep='|', skip_blank_lines = False)
    nameIndices = {x:i for i,x in enumerate(names['NAME'])}
    partyIndices = {'DEM':0, 'REP':1, 'OTHERS':2}
    N = names.shape[0]

    df = pandas.read_csv(path + filename, sep = '|', skip_blank_lines = False)
    df = df.drop(df[df.DONATION <= 0].index)

    df['NAME_I'] = df['NAME'].map(nameIndices)
    df['PARTY_J'] = df['PARTY'].map(partyIndices)   
    df = df[df.NAME_I.notnull()]
    df = df[df.PARTY_J.notnull()]
    donations = np.array(sparse.csc_matrix((df['DONATION'], (df['NAME_I'], df['PARTY_J'])), shape = (N, 3)).toarray())
    return donations

def mapIdToNames(path, filename):
    names = pandas.read_csv(path + elecion_files[0], sep='|', skip_blank_lines = False)
    nameIndices = {x:i for i,x in enumerate(names['NAME'])}
    partyIndices = {'DEM':0, 'REP':1, 'OTHERS':2}
    N = names.shape[0]
    df = pandas.read_csv(path + filename, sep = '|', skip_blank_lines = False)
    df = df.drop(df[df.DONATION <= 0].index)
    df['NAME_I'] = df['NAME'].map(nameIndices)
    df['PARTY_J'] = df['PARTY'].map(partyIndices)    
    df = df[df.NAME_I.notnull()]
    df = df[df.PARTY_J.notnull()]
    donations = np.array(sparse.csc_matrix((df['DONATION'], (df['NAME_I'], df['PARTY_J'])), shape = (N, 3)).toarray())
    idToNames = {}
    ids, names = list(df['NAME_I']), list(df['NAME'])
    for i, id in enumerate(ids):
        if id > 0 and id in idToNames and not idToNames[id] == names[i]:
            logging.warn('Inconsistent mapping found: df[NAME_I] to df[NAME]: ' + str(id) + ': ' + str(idToNames[id]) + ', ' + str(names[i]))
        idToNames[id] = names[i]
    return idToNames, donations

def generateReport(nodeIdContributorsList):
    path = election_data_path
    idToNames, donations = mapIdToNames(path, elecion_files[1])
    donations16 = loadContributionsCSV(path, elecion_files[2])
    print '\n ========== \n ***** report ******* \n ======== \n'
    for row in nodeIdContributorsList:
        id, contributors = row[0], row[1:]
        print 'Ids:   ', id, contributors
        print 'Names: ', idToNames.get(id), [idToNames.get(x) for x in contributors]
        print '2012:  ', donations[id], [donations[x] for x in contributors]
        print '2016:  ', donations16[id], [donations16[x] for x in contributors]
        print '----------'
    print '\n ========== \n ***** report END ******* \n ======== \n'
    return

def electionEdgesReport(topAnomalyEdges):
    path = election_data_path
    idToNames, donations12 = mapIdToNames(path, elecion_files[1])
    donations16 = loadContributionsCSV(path, elecion_files[2])
    print '\n\n\n  ', topAnomalyEdges
    print '\n ========== \n ***** report ******* \n ======== \n'
    for score, (i, j) in topAnomalyEdges:
        print idToNames.get(i), ' -- ', idToNames.get(j), '  s = ', score
        print '2012:  ', donations12[i], donations12[j]
        print '2016:  ', donations16[i], donations16[j], '\n'
    print '\n ========== \n ***** report END ******* \n ======== \n'
    return

