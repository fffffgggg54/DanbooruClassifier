import pandas as pd
from multiprocessing import Pool
import random
from itertools import islice

# json loading methods

# SIMD function to import each chunk as a pandas DataFrame
def readFile(lineList):
    return pd.read_json(''.join(lineList), lines=True)

# parallel json importer
# TODO profile and optimize speed and memory usage, reduce serial tasks
# TODO rewrite multithreaded routines to make the main thread check unnecesary

# https://stackoverflow.com/a/46941313
def dataImporter(filePath, linesPerChunk = 1000, importThreadCount = 30, keep = 1.0):
    ingestChunks = []
    importedChunks = []
    
    print("Reading " + filePath)

    # split file into chunks

    with open(filePath, 'r', encoding = 'utf-8') as currFile:
        lines = []
        for line in islice(currFile, 0, None, int(1/keep)):
            lines.append(line)
        #lines = currFile.readlines()
        lineCount = len(lines)
        print("Got " + str(len(lines)) + " lines")
        lines = [lines[startLine : startLine + linesPerChunk] for startLine in range(0, lineCount, int(linesPerChunk))]
        
        
        #ingestChunks.extend([''.join(lines[startLine : startLine + linesPerChunk]) for startLine in range(0, lineCount, linesPerChunk)])
        
        print("Got " + str(len(lines)) + " chunks of " + str(linesPerChunk) + ", parsing")

    #print(ingestChunks[0])
    #exit()
    
    
    # TODO investigate reliability of this - single worker occasionally hangs
    # TODO investigate high comitted memory usage

    with Pool(importThreadCount) as pool:
        importedChunks = pool.map(readFile, lines)
        pool.terminate()
        pool.join()
        pool.close()
    
    result = pd.concat(importedChunks, ignore_index = True)
    print("Imported " + str(len(importedChunks)) + " chunks")
    #print(result)
    return result
