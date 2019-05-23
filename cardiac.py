#!/usr/bin/env python
from __future__ import print_function

import matplotlib
matplotlib.use('agg')

import json
import numpy as np
import os
import pints
import sys

from collections import OrderedDict

# Imports required to construct manifest file for Web Lab
from xml.etree.ElementTree import ElementTree, Element, SubElement

# Modeling framework imports
from modeling.distributions import (
    DiscreteParameterDistribution,
    IndependentParameterDistribution,
    Uniform,
)
from modeling.objective import LogLikGauss
from modeling.algorithm import ParameterFittingTask
from modeling.io import ReadDataSet, ReadParameterDistribution

# TODO: This script, called by the server, doesn't know where to find fc, or
# things like the CompactSyntaxParser, that are required to run simulations...
CHASTE_ROOT = os.environ['CHASTE_ROOT']
FC_ROOT = os.environ['FC_ROOT']

sys.path.append(os.path.join(FC_ROOT, 'src', 'python'))
sys.path.append(os.path.join(FC_ROOT, 'src', 'proto', 'parsing'))
sys.path.append(os.path.join(CHASTE_ROOT, 'python', 'pycml'))
from modeling.fcexperiment import FunctionalCurationExperiment


def readFittingProtocol(jsonProtoFile):
    """
    Read a JSON-specified fitting protocol containing the following entries:
     - 'algorithm': string
     - 'arguments': dict/JSON
     - 'output': dict/JSON
     - 'prior': dict/JSON
    """
    pf = open(jsonProtoFile)

    # HACK HACK HACK
    # Using ordered dict to preserve order, but this does NOT follow the JSON
    # standard
    protoDict = json.load(pf, object_pairs_hook=OrderedDict)

    if not isinstance(protoDict.get('prior', None), dict):
        raise ValueError(
            'Prior distribution must be specified as either name/value or'
            ' name/tuple pairs')
    for pname, value in protoDict['prior'].iteritems():
        if (not isinstance(value, list)) or len(value) != 2:
            raise ValueError('Incorrect specification of prior')
    if not 'output' in protoDict:
        raise ValueError('One or more named outputs must be provided')

    parameters = list(protoDict['prior'].keys())

    prior = makePriorFromDict(protoDict['prior'])

    optimisation_algorithm = protoDict.get(
        'optimisation_algorithm', 'CMAES')
    optimisation_arguments = protoDict.get(
        'optimisation_arguments', {'repeats': 0})

    sampling_algorithm = protoDict.get(
        'sampling_algorithm', 'AdaptiveCovarianceMCMC')
    sampling_arguments = protoDict.get(
        'sampling_arguments', {'iterations': 0})
    n_mcmc = sampling_arguments.get('iterations', 0)
    if n_mcmc:
        n_warm = sampling_arguments.get('warm_up', 0)
        if n_warm >= n_mcmc:
            raise ValueError(
                'If MCMC is used, the number of warm-up iterations must be'
                ' smaller than the number of MCMC iterations.')

    outputs = protoDict['output']

    # Optional specification of protocol inputs
    inputs = None
    if 'input' in protoDict:
        inputs = protoDict['input']

    # Standard deviation of noise
    noise_std = float(protoDict.get('noise_std', 1.0))

    return (
        parameters,
        prior,
        outputs,
        optimisation_algorithm,
        optimisation_arguments,
        sampling_algorithm,
        sampling_arguments,
        inputs,
        noise_std,
    )


def makePriorFromDict(priorDict):
    """
    Generates an IndependentParameterDistribution object from a dictionary of
    name: (float1, float2)
    """
    distributions = {}
    for pname, value in priorDict.iteritems():
        distributions[str(pname)] = Uniform(value[0], value[1])
    return IndependentParameterDistribution(distributions)


def run_fit(
        task,
        optimisation_algorithm,
        optimisation_arguments,
        sampling_algorithm,
        sampling_arguments,
        ):

    print('Entering run_fit()')

    if optimisation_algorithm != 'CMAES':
        raise ValueError(
            'Other optimisation algorithms are not yet supported.')
    if sampling_algorithm != 'AdaptiveCovarianceMCMC':
        raise ValueError(
            'Other sampling algorithms are not yet supported.')

    # Get names of parameters --> They are not stored in order, so will need
    # this a lot!
    keys = task.parameters

    # Use log transform
    log_transform = task.prior.is_positive()
    if log_transform:
        print('Using log-transform for fitting')
    else:
        print('Unable to use log-transform')

    # Select objective for Aidan's code to use
    task.objFun = LogLikGauss()

    # Wrap a LogPDF around Aidan's objective
    class AidanLogPdf(pints.LogPDF):
        def __init__(self, task, keys, log_transform=None):
            self._task = task
            self._keys = keys
            self._log_transform = log_transform
            self._dimension = len(keys)
            self._p = {}

        def n_parameters(self):
            return self._dimension

        def __call__(self, x):

            # Untransform back to model space
            if self._log_transform:
                x = np.exp(x)

            # Create dict
            for i, key in enumerate(self._keys):
                self._p[key] = x[i]

            # Evaluate objective
            return self._task.calculateObjective(self._p)

    # Wrap a LogPrior around Aidan's prior
    class AidanLogPrior(pints.LogPrior):
        def __init__(self, task, keys, log_transform=None):
            self._prior = task.prior
            self._keys = keys
            self._log_transform = log_transform
            self._dimension = len(keys)
            self._p = {}

        def n_parameters(self):
            return self._dimension

        def __call__(self, x):

            # Untransform back to model space
            if self._log_transform:
                x = np.exp(x)

            # Create dict
            for i, key in enumerate(self._keys):
                self._p[key] = x[i]

            # Evaluate prior and return
            prior = self._prior.pdf(self._p)
            if prior <= 0:
                return -np.inf
            return np.log(prior)

        def sample(self, n=1):

            assert n == 1
            x = self._prior.draw()
            x = [x[key] for key in self._keys]

            # Transform to search space
            if self._log_transform:
                x = np.log(x)

            return [x]

    # Find a suitable starting point --> Will be the answer if no iterations
    # are selected
    log_prior = AidanLogPrior(task, keys, False)
    x0 = log_prior.sample()[0]
    del(log_prior)

    print('Parameters: ')
    print('\n'.join('  ' + x for x in parameters))

    # If specified, run (repeated) CMA-ES to select a starting point
    opt_repeats = optimisation_arguments['repeats']
    print('CMA-ES runs: ' + str(opt_repeats))
    if opt_repeats:

        log_likelihood = AidanLogPdf(task, keys, log_transform)
        boundaries = pints.LogPDFBoundaries(
            AidanLogPrior(task, keys, log_transform))

        x_best, fx_best = x0, -np.inf

        for i in range(opt_repeats):

            print(' CMA-ES run ' + str(1 + i))

            # Choose random starting point (in search space)
            x0 = boundaries.sample()[0]
            f0 = log_likelihood(x0)
            i = 0
            while not np.isfinite(f0):
                x0 = boundaries.sample()[0]
                f0 = log_likelihood(x0)
                i += 1
                if i > 20:
                    print('Unable to find good starting point!')
                    break

            # Create optimiser
            opt = pints.OptimisationController(
                log_likelihood, x0, boundaries=boundaries, method=pints.CMAES)
            opt.set_max_iterations(None)
            opt.set_parallel(True)
            opt.set_max_unchanged_iterations(80)

            # DEBUG
            #opt.set_max_iterations(5)

            # Run optimisation
            try:
                with np.errstate(all='ignore'):
                    x, fx = opt.run()
            except ValueError:
                fx = -np.inf
                import traceback
                traceback.print_exc()

            # Check outcome
            if fx > fx_best:
                print('New best score ' + str(fx) + ' > ' + str(fx_best))
                x_best, fx_best = x, fx

                if log_transform:
                    # Transform back to model space
                    x_best = np.exp(x_best)

        x0 = x_best
        x0_obj = dict(zip(keys, x0))

    # If specified, run MCMC
    n_mcmc_iters = sampling_arguments.get('iterations', 0)
    print('MCMC iterations: ' + str(n_mcmc_iters))
    if n_mcmc_iters:
        print('Starting MCMC')

        log_likelihood = AidanLogPdf(task, keys)
        log_prior = AidanLogPrior(task, keys)
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)

        # Configure MCMC
        mcmc = pints.MCMCSampling(log_posterior, 1, [x0])
        mcmc.set_max_iterations(n_mcmc_iters)
        mcmc.set_parallel(False)

        # Run
        chains = mcmc.run()
        print('MCMC Completed')

        # Get chain
        chain = chains[0]

        # Discard warm-up
        warm_up = int(sampling_arguments.get('warm_up', 0))
        if warm_up > 0:
            print('Discarding first ' + str(warm_up) + ' samples as warm-up')
            chain = chain[warm_up:]

    else:

        chain = [x0]

    # Create distribution object and return
    dist = []
    for sample in chain:
        d = {}
        for i, key in enumerate(keys):
            d[key] = sample[i]
        dist.append(d)
    return DiscreteParameterDistribution(dist)


def createFittingTask(
        parameters, prior, modelFile, simProto, dataFile, outputs, inputs,
        noise_std,
        ):
    # Create experiment
    experiment = FunctionalCurationExperiment(protoFile, modelFile)

    # Read data file
    dataSet = ReadDataSet(dataFile)

    # Set inputs (if provided)
    if inputs != None:
        for inputName, key in inputs.iteritems():
            if key not in dataSet:
                raise ValueError(
                    'Inputs must specify a valid column in the data file')
            # TODO: Make sure inputName is a valid protocol input
            experiment.setInputs({inputName:dataSet[key]})

    # Check all outputs are valid
    for outputName, key in outputs.iteritems():
        if key not in dataSet:
            raise ValueError(
                'Outputs must specify a valid column in the data file')
        # TODO: Make sure outputName is a valid protocol output

    return ParameterFittingTask(
        parameters,
        prior,
        experiment,
        dataSet,
        outputs,
        {'noise_std': noise_std},
    )


def writeOutputs(
        parameters, meanParams, expDataFile, simData, outputs, outputDir,
        manifest, contsMeta, plotsMeta,
        ):
    expData = ReadDataSet(expDataFile)

    # Write a CSV with the obtained (mean) parameters
    fname = 'obtained_parameters.csv'
    with open(os.path.join(outputDir, fname), 'w') as f:
        keys = parameters
        ks = ['"' + key + '"' for key in keys]
        vs = ['{:< 1.17e}'.format(meanParams[key]) for key in keys]
        f.write(','.join(ks) + '\n')
        f.write(','.join(vs) + '\n')
    SubElement(manifest, 'content', location='/' + fname, format='text/csv')

    # Write a CSV with labels for each trace
    with open(os.path.join(outputDir, "outputs_trace_labels.csv"), 'w') as f:
        f.write('# Labels\n1,2\n"Experimental data"\n"Posterior mean"\n')
    SubElement(
        manifest,
        "content",\
        location="/outputs_trace_labels.csv",
        format="text/csv")

    # Write CSV data files for each simulated and observed output, along with
    # one for comparative plotting
    for protoOutputName, expDataName in outputs.iteritems():
        simDataFileName = "outputs_" + protoOutputName + "_mean_predicted.csv"
        expDataFileName = "outputs_" + expDataName + "_observed.csv"
        pltDataFileName = "outputs_" + protoOutputName + "_gnuplot_data.csv"

        simDataFile = open(os.path.join(outputDir, simDataFileName), "w+")
        expDataFile = open(os.path.join(outputDir, expDataFileName), "w+")
        pltDataFile = open(os.path.join(outputDir, pltDataFileName), "w+")
        indices = open(os.path.join(outputDir, "index.csv"),"w+")

        npts = len(expData[expDataName])

        # TODO: Implicitly requires 1- or 0-d data, which is what we currently
        # support. Should we expand our scope, this will have to change.
        simDataFile.write("# " + protoOutputName + "\n")
        simDataFile.write("1, " + str(npts) + "\n")
        expDataFile.write("# " + expDataName + "\n")
        expDataFile.write("1, " + str(npts) + "\n")
        indices.write("# Indices\n1, "+str(npts) + "\n")

        for i in range(npts):
            simDataFile.write(str(simData[protoOutputName][i]) + '\n')
            expDataFile.write(str(expData[expDataName][i]) + '\n')
            pltDataFile.write(
                str(i) + ',' + str(expData[expDataName][i]) + ',' +
                str(simData[protoOutputName][i]) + '\n')
            indices.write(str(i) + "\n")

        simDataFile.close()
        expDataFile.close()
        pltDataFile.close()
        indices.close()

        SubElement(
            manifest,"content",location="/" + simDataFileName, format="text/csv")
        SubElement(
            manifest,"content",location="/" + expDataFileName, format="text/csv")
        SubElement(
            manifest,"content",location="/" + pltDataFileName, format="text/csv")
        SubElement(
            manifest,"content",location="/index.csv", format="text/csv")

        # Add entries to metadata files to allow for plotting
        fh = open(contsMeta,"a")
        fh.write("index,,Index,1,index.csv,raw," + str(npts) + "\n")
        fh.write(
            protoOutputName + "_mean_predicted," + protoOutputName
            + ",units,1,outputs_" + protoOutputName
            + "_mean_predicted.csv,raw," + str(npts) + "\n")
        fh.write("labels,Trace,,1,outputs_trace_labels.csv,raw,2\n")
        fh.close()

        fh = open(plotsMeta,"a")
        fh.write(
            str(protoOutputName) + ",," + pltDataFileName + ",lines,index,"
            + protoOutputName + "_mean_predicted,labels\n")

    print('Wrote plottable outputs')


def writePosterior(dist, outputDir, manifest, contsMeta, plotsMeta):
    try:
        marginals = dist._marginals()
        weights = dist.weights

        for pname in marginals:
            writeHisto(
                pname,
                marginals[pname],
                weights,
                outputDir,
                manifest,
                contsMeta,
                plotsMeta)

    except:
        print('Parameter distribution not of correct form')


def writeHisto(name, vals, weights, outputDir, manifest, contsMeta, plotsMeta):
    binHeights,binLocs = np.histogram(vals, bins=10, weights=weights)
    bins = zip(binLocs,binHeights)

    if ':' in name:
        pname = name.split(':')[1]
    else:
        pname = name

    fh = os.path.join(outputDir,pname + "_histogram_gnuplot_data.csv")
    binFile = os.path.join(outputDir, "outputs_" + pname) + "_bins.csv"
    freqFile = os.path.join(outputDir, "outputs_" + pname) + "_freq.csv"

    try:
        fh = open(fh, "w+")
        binFile = open(binFile, "w+")
        freqFile = open(freqFile, "w+")

        binFile.write("1," + str(len(bins)) + "\n")
        freqFile.write("1," + str(len(bins)) + "\n")
        for entry in bins:
            fh.write(str(entry[0]) + "," + str(entry[1] * 100) + "\n")
            binFile.write(str(entry[0]) + "\n")
            freqFile.write(str(entry[1] * 100) + "\n")

        # Add output files to manifest
        SubElement(
            manifest,
            "content",
            location="/" + pname + "_histogram_gnuplot_data.csv",
            format="text/csv")
        SubElement(
            manifest,
            "content",
            location="/outputs_" + pname + "_bins.csv",
            format="text/csv")
        SubElement(
            manifest,
            "content",
            location="/outputs_" + pname + "_freq.csv",
            format="text/csv")

        # Add output information to metadata files, to allow for correct
        # plotting/annotation
        fh.close()
        fh = open(contsMeta, "a")
        # TODO: Replace 'units' with actual units, supplied as an optional
        # argument. For FC case, would have to be able to look up the units of
        # each parameter
        fh.write(
            pname + "_bins," + pname + ",units,1,outputs_" + pname
            + "_bins.csv,raw," + str(len(bins)) + "\n")
        fh.write(
            pname + "_freq,Frequency,%,outputs_" + pname + "_freq.csv,raw,"
            + str(len(bins)) + "\n")

        fh = open(plotsMeta,"a")
        fh.write(
            "Post prob,," + pname + "_histogram_gnuplot_data.csv,hist," + pname
            + "_bins," + pname + "_freq\n")
    finally:
        fh.close()
        if isinstance(binFile, file):
            binFile.close()
        if isinstance(freqFile, file):
            freqFile.close()


if __name__ == "__main__":
    # Must supply a full fitting specification
    assert len(sys.argv) == 6

    modelFile = sys.argv[1]  # CellML model file
    protoFile = sys.argv[2]  # Functional curation protocol file
    fitProto = sys.argv[3]   # Fitting protocol file
    dataFile = sys.argv[4]   # Experimental data file
    outputDir = sys.argv[5]  # Output directory

    top = Element(
        "omexManifest",
        xmlns="http://identifiers.org/combine.specifications/omex-manifest")
    SubElement(
        top,
        "content",
        location="/manifest.xml",
        format="http://identifiers.org/combine.specifications/omex-manifest")

    # Create tmp directory, if it doesn't already exist
    if not os.path.isdir(outputDir) or not os.path.exists(outputDir):
        try:
            # NOTE: For some reason, the FC executable constructs tmp
            # directories that are 3 levels deep. Because this structure is
            # expected by tasks.py for simulation results, [Aidan has] kept it
            # the same here for fitting results.
            os.mkdir(outputDir)
            outputDir = os.path.join(outputDir, "1")
            os.mkdir(outputDir)
            outputDir = os.path.join(outputDir, "2")
            os.mkdir(outputDir)
        except:
            print('Could not create temporary directory')

    SubElement(top, "content", location="/stdout.txt", format="text/plain")

    (
        parameters,
        prior,
        outputs,
        optimisation_algorithm,
        optimisation_arguments,
        sampling_algorithm,
        sampling_arguments,
        inputs,
        noise_std,
    ) = readFittingProtocol(fitProto)

    # Store current working directory -- need to change to CHASTE_ROOT to
    # execute fitting, and will revert after
    orig_cwd = os.getcwd()
    os.chdir(CHASTE_ROOT)

    task = createFittingTask(
        parameters, prior, modelFile, protoFile, dataFile, outputs, inputs,
        noise_std
    )

    post = run_fit(
        task,
        optimisation_algorithm,
        optimisation_arguments,
        sampling_algorithm,
        sampling_arguments,
    )

    # Revert to previous working directory once fitting is over (not sure if
    # needed, but to avoid confusion later!)
    os.chdir(orig_cwd)

    # Generate plots
    contsMetaFile = os.path.join(outputDir,"outputs-contents.csv")
    contsMeta = open(contsMetaFile,'w+')
    contsMeta.write(','.join([
        'Variable id',
        'Variable name',
        'Units',
        'Number of dimensions',
        'File name',
        'Type',
        'Dimensions',
    ]))
    contsMeta.write("\n")
    contsMeta.close()

    plotsMetaFile = os.path.join(outputDir, 'outputs-default-plots.csv')
    plotsMeta = open(plotsMetaFile,'w+')
    plotsMeta.write(','.join([
        'Plot title',
        'File name',
        'Data file name',
        'Line style',
        'First variable id',
        'Second variable id',
        'Optional key variable id',
    ]))
    plotsMeta.write("\n")
    plotsMeta.close()

    # Histograms for posterior distributions
    if sampling_arguments.get('iterations', 0):
        writePosterior(post, outputDir, top, contsMetaFile, plotsMetaFile)

    # Experimental data vs. mean predicted data for each output
    meanParams = post.mean()
    meanSimData = task.experiment.simulate(meanParams)
    writeOutputs(
        parameters,
        meanParams,
        dataFile,
        meanSimData,
        outputs,
        outputDir,
        top,
        contsMetaFile,
        plotsMetaFile)

    SubElement(
        top,
        "content",
        location="/outputs-contents.csv",
        format="text/csv")
    SubElement(
        top,
        "content",
        location="/outputs-default-plots.csv",
        format="text/csv")
    print('Generated plots')

    # Create "success" file to let website know it worked
    success = open(os.path.join(outputDir, "success"),"w+")
    success.write("Protocol completed successfully")
    success.close()
    SubElement(top,"content", location="/success", format="text/plain")

    print('Wrote success file')

    # Generate the manifest file
    manifest = ElementTree(top)
    manifest.write(
        os.path.join(outputDir, "manifest.xml"),
        encoding="UTF-8",
        xml_declaration=True)

    print('Wrote manifest')

    #print(os.listdir(outputDir))

    print('Done')

