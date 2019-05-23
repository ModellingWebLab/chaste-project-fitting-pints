from __future__ import absolute_import
import csv

from . import distributions


def ReadParameterDistribution(fileName,delim='\t',weightCol=None):
    '''
    Reads a discrete parameter distribution from a file containing
    newline-separated delimited parameter vectors.
    - Parameter names specified in first row
    - Weights (optionally) specified by a column index
    - Supports headers commented by "#"
    '''

    particles,weights = [],[]

    csvfile = open(fileName)
    reader = csv.reader(csvfile,delimiter=delim)

    pnames = reader.next()
    while pnames[0][0] == '#':
        pnames = reader.next()

    if "WEIGHTS" in pnames:
        weightCol = pnames.index("WEIGHTS")
    if weightCol == None:
        weights = None

    for line in reader:
        try:
            particle = {}
            for i,p in enumerate(pnames):
                if i != weightCol:
                    particle[p] = float(line[i])
                else:
                    weights.append(float(line[i]))
            particles.append(particle)
        except:
            pass

    csvfile.close()
    return distributions.DiscreteParameterDistribution(particles, weights)


def ReadDataSet(fileName,delim='\t'):
    '''
    Read one or more 0- or 1-D numerical outputs into a dict
    '''

    csvfile = open(fileName)

    reader = csv.reader(csvfile,delimiter=delim)
    outputs = reader.next()

    while outputs[0][0] == "#":
        outputs = reader.next()

    dataSet = dict([(key,[]) for key in outputs])

    for row in reader:
        for i,entry in enumerate(row):
            dataSet[outputs[i]].append(float(entry))

    return dataSet

