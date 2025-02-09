using System;
using System.Diagnostics;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.GradientBoost.GBMDecisionTree;
using SharpLearning.GradientBoost.Loss;
using SharpLearning.GradientBoost.Models;

namespace SharpLearning.GradientBoost.Learners;

/// <summary>
/// Classification gradient boost learner based on 
/// http://statweb.stanford.edu/~jhf/ftp/trebst.pdf
/// A series of regression trees are fitted stage wise on the residuals of the previous stage.
/// The resulting models are ensembled together using addition. Implementation based on:
/// http://gradientboostedmodels.googlecode.com/files/report.pdf
/// </summary>
public class ClassificationGradientBoostLearner
    : IIndexedLearner<double>
    , IIndexedLearner<ProbabilityPrediction>
    , ILearner<double>
    , ILearner<ProbabilityPrediction>
{
    readonly GBMDecisionTreeLearner m_learner;
    readonly double m_learningRate;
    readonly int m_iterations;
    readonly double m_subSampleRatio;
    readonly Random m_random = new(42);
    readonly IGradientBoostLoss m_loss;

    /// <summary>
    ///  Base classification gradient boost learner. 
    ///  A series of regression trees are fitted stage wise on the residuals of the previous stage
    /// </summary>
    /// <param name="iterations">The number of iterations or stages</param>
    /// <param name="learningRate">How much each iteration should contribute with</param>
    /// <param name="maximumTreeDepth">The maximum depth of the tree models</param>
    /// <param name="minimumSplitSize">minimum node split size in the trees 1 is default</param>
    /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
    /// <param name="subSampleRatio">ratio of observations sampled at each iteration. Default is 1.0. 
    /// If below 1.0 the algorithm changes to stochastic gradient boosting. 
    /// This reduces variance in the ensemble and can help outer overfitting</param>
    /// <param name="featuresPrSplit">Number of features used at each split in the tree. 0 means all will be used</param>
    /// <param name="loss">loss function used</param>
    /// <param name="runParallel">Use multi threading to speed up execution</param>
    public ClassificationGradientBoostLearner(
        int iterations,
        double learningRate,
        int maximumTreeDepth,
        int minimumSplitSize,
        double minimumInformationGain,
        double subSampleRatio,
        int featuresPrSplit,
        IGradientBoostLoss loss,
        bool runParallel)
    {
        if (iterations < 1) { throw new ArgumentException("Iterations must be at least 1"); }
        if (learningRate <= 0.0) { throw new ArgumentException("learning rate must be larger than 0"); }
        if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
        if (maximumTreeDepth < 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
        if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
        if (subSampleRatio <= 0.0 || subSampleRatio > 1.0) { throw new ArgumentException("subSampleRatio must be larger than 0.0 and at max 1.0"); }
        if (featuresPrSplit < 0) { throw new ArgumentException("featuresPrSplit must be at least 0"); }
        m_loss = loss ?? throw new ArgumentNullException(nameof(loss));

        m_iterations = iterations;
        m_learningRate = learningRate;
        m_subSampleRatio = subSampleRatio;
        m_learner = new GBMDecisionTreeLearner(maximumTreeDepth, minimumSplitSize,
            minimumInformationGain, featuresPrSplit, m_loss, runParallel);
    }

    /// <summary>
    ///  Base classification gradient boost learner. 
    ///  A series of regression trees are fitted stage wise on the residuals of the previous stage
    /// </summary>
    /// <param name="iterations">The number of iterations or stages</param>
    /// <param name="learningRate">How much each iteration should contribute with</param>
    /// <param name="maximumTreeDepth">The maximum depth of the tree models</param>
    /// <param name="minimumSplitSize">minimum node split size in the trees 1 is default</param>
    /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
    /// <param name="subSampleRatio">ratio of observations sampled at each iteration. Default is 1.0. 
    /// If below 1.0 the algorithm changes to stochastic gradient boosting. 
    /// This reduces variance in the ensemble and can help counter overfitting</param>
    /// <param name="featuresPrSplit">Number of features used at each split in the tree. 0 means all will be used</param>
    public ClassificationGradientBoostLearner(
        int iterations = 100,
        double learningRate = 0.1,
        int maximumTreeDepth = 3,
        int minimumSplitSize = 1,
        double minimumInformationGain = 0.000001,
        double subSampleRatio = 1.0,
        int featuresPrSplit = 0)
        : this(iterations, learningRate, maximumTreeDepth, minimumSplitSize, minimumInformationGain,
            subSampleRatio, featuresPrSplit, new GradientBoostBinomialLoss(), true)
    {
    }

    /// <summary>
    ///  A series of regression trees are fitted stage wise on the residuals of the previous stage
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    public ClassificationGradientBoostModel Learn(F64Matrix observations, double[] targets)
    {
        var allIndices = Enumerable.Range(0, targets.Length).ToArray();
        return Learn(observations, targets, allIndices);
    }

    /// <summary>
    ///  A series of regression trees are fitted stage wise on the residuals of the previous stage
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    public ClassificationGradientBoostModel Learn(F64Matrix observations, double[] targets,
        int[] indices)
    {
        Checks.VerifyObservationsAndTargets(observations, targets);
        Checks.VerifyIndices(indices, observations, targets);

        var rows = observations.RowCount;
        var orderedElements = CreateOrderedElements(observations, rows);

        var inSample = targets.Select(t => false).ToArray();
        indices.ForEach(i => inSample[i] = true);
        var workIndices = indices.ToArray();

        var uniqueTargets = targets.Distinct().OrderBy(v => v).ToArray();
        var initialLoss = m_loss.InitialLoss(targets, inSample);

        double[][] oneVsAllTargets = null;
        double[][] predictions = null;
        double[][] residuals = null;
        GBMTree[][] trees = null;

        if (uniqueTargets.Length == 2) // Binary case - only need to fit to one class and use (1.0 - probability)
        {
            trees = new GBMTree[][] { new GBMTree[m_iterations] };
            predictions = new double[][] { targets.Select(_ => initialLoss).ToArray() };
            residuals = new double[][] { new double[targets.Length] };

            oneVsAllTargets = new double[1][];
            var target = uniqueTargets[0];
            oneVsAllTargets[0] = targets.Select(t => t == target ? 1.0 : 0.0).ToArray();

        }
        else // multi-class case - use oneVsAll strategy and fit probability for each class
        {
            trees = new GBMTree[uniqueTargets.Length][];
            predictions = uniqueTargets.Select(_ => targets.Select(t => initialLoss).ToArray())
                .ToArray();
            residuals = uniqueTargets.Select(_ => new double[targets.Length])
                .ToArray();

            oneVsAllTargets = new double[uniqueTargets.Length][];
            for (var i = 0; i < uniqueTargets.Length; i++)
            {
                var target = uniqueTargets[i];
                oneVsAllTargets[i] = targets.Select(t => t == target ? 1.0 : 0.0).ToArray();
                trees[i] = new GBMTree[m_iterations];
            }
        }

        var predictWork = new double[observations.RowCount];
        for (var iteration = 0; iteration < m_iterations; iteration++)
        {
            for (var itarget = 0; itarget < trees.Length; itarget++)
            {
                m_loss.UpdateResiduals(oneVsAllTargets[itarget], predictions[itarget],
                    residuals[itarget], inSample);

                var sampleSize = targets.Length;
                if (m_subSampleRatio != 1.0)
                {
                    sampleSize = (int)Math.Round(m_subSampleRatio * workIndices.Length);
                    var currentInSample = Sample(sampleSize, workIndices, targets.Length);

                    trees[itarget][iteration] = m_learner.Learn(observations, oneVsAllTargets[itarget],
                        residuals[itarget], predictions[itarget], orderedElements, currentInSample);
                }
                else
                {
                    trees[itarget][iteration] = m_learner.Learn(observations, oneVsAllTargets[itarget],
                        residuals[itarget], predictions[itarget], orderedElements, inSample);
                }

                trees[itarget][iteration].Predict(observations, predictWork);
                for (var i = 0; i < predictWork.Length; i++)
                {
                    predictions[itarget][i] += m_learningRate * predictWork[i];
                }
            }
        }

        return new ClassificationGradientBoostModel(trees, uniqueTargets, m_learningRate,
            initialLoss, observations.ColumnCount);
    }

    /// <summary>
    /// Learns a ClassificationGradientBoostModel with early stopping.
    /// The parameter earlyStoppingRounds controls how often the validation error is measured.
    /// If the validation error has increased, the learning is stopped and the model with the best number of iterations (trees) is returned.
    /// The number of iterations used is equal to the number of trees in the resulting model.
    /// The method used for early stopping is based on the article:
    /// http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
    /// </summary>
    /// <param name="trainingObservations"></param>
    /// <param name="trainingTargets"></param>
    /// <param name="validationObservations"></param>
    /// <param name="validationTargets"></param>
    /// <param name="metric">The metric to use for early stopping</param>
    /// <param name="earlyStoppingRounds">This controls how often the validation error is checked to estimate the best number of iterations.</param>
    /// <returns>ClassificationGradientBoostModel with early stopping. The number of iterations will equal the number of trees in the model</returns>
    public ClassificationGradientBoostModel LearnWithEarlyStopping(
        F64Matrix trainingObservations,
        double[] trainingTargets,
        F64Matrix validationObservations,
        double[] validationTargets,
        IMetric<double, double> metric,
        int earlyStoppingRounds)
    {
        if (earlyStoppingRounds >= m_iterations)
        {
            throw new ArgumentException("Number of iterations " + m_iterations +
                " is smaller than earlyStoppingRounds " + earlyStoppingRounds);
        }

        Checks.VerifyObservationsAndTargets(trainingObservations, trainingTargets);
        Checks.VerifyObservationsAndTargets(validationObservations, validationTargets);

        var rows = trainingObservations.RowCount;
        var orderedElements = CreateOrderedElements(trainingObservations, rows);

        var inSample = trainingTargets.Select(t => false).ToArray();
        var indices = Enumerable.Range(0, trainingTargets.Length).ToArray();
        indices.ForEach(i => inSample[i] = true);
        var workIndices = indices.ToArray();

        var uniqueTargets = trainingTargets.Distinct().OrderBy(v => v).ToArray();
        var initialLoss = m_loss.InitialLoss(trainingTargets, inSample);

        double[][] oneVsAllTargets = null;
        double[][] predictions = null;
        double[][] residuals = null;
        GBMTree[][] trees = null;

        if (uniqueTargets.Length == 2) // Binary case - only need to fit to one class and use (1.0 - probability)
        {
            trees = new GBMTree[][] { new GBMTree[m_iterations] };
            predictions = new double[][] { trainingTargets.Select(_ => initialLoss).ToArray() };
            residuals = new double[][] { new double[trainingTargets.Length] };

            oneVsAllTargets = new double[1][];
            var target = uniqueTargets[0];
            oneVsAllTargets[0] = trainingTargets.Select(t => t == target ? 1.0 : 0.0).ToArray();

        }
        else // multi-class case - use oneVsAll strategy and fit probability for each class
        {
            trees = new GBMTree[uniqueTargets.Length][];
            predictions = uniqueTargets.Select(_ => trainingTargets.Select(t => initialLoss).ToArray())
                .ToArray();
            residuals = uniqueTargets.Select(_ => new double[trainingTargets.Length])
                .ToArray();

            oneVsAllTargets = new double[uniqueTargets.Length][];
            for (var i = 0; i < uniqueTargets.Length; i++)
            {
                var target = uniqueTargets[i];
                oneVsAllTargets[i] = trainingTargets.Select(t => t == target ? 1.0 : 0.0).ToArray();
                trees[i] = new GBMTree[m_iterations];
            }
        }

        var bestIterationCount = 0;
        var currentBedstError = double.MaxValue;

        var predictWork = new double[trainingObservations.RowCount];
        for (var iteration = 0; iteration < m_iterations; iteration++)
        {
            for (var itarget = 0; itarget < trees.Length; itarget++)
            {
                m_loss.UpdateResiduals(oneVsAllTargets[itarget], predictions[itarget],
                    residuals[itarget], inSample);

                var sampleSize = trainingTargets.Length;
                if (m_subSampleRatio != 1.0)
                {
                    sampleSize = (int)Math.Round(m_subSampleRatio * workIndices.Length);
                    var currentInSample = Sample(sampleSize, workIndices, trainingTargets.Length);

                    trees[itarget][iteration] = m_learner.Learn(trainingObservations, oneVsAllTargets[itarget],
                        residuals[itarget], predictions[itarget], orderedElements, currentInSample);
                }
                else
                {
                    trees[itarget][iteration] = m_learner.Learn(trainingObservations, oneVsAllTargets[itarget],
                        residuals[itarget], predictions[itarget], orderedElements, inSample);
                }

                trees[itarget][iteration].Predict(trainingObservations, predictWork);
                for (var i = 0; i < predictWork.Length; i++)
                {
                    predictions[itarget][i] += m_learningRate * predictWork[i];
                }
            }

            // When using early stopping, Check that the validation error is not increasing between earlyStoppingRounds
            // If the validation error has increased, stop the learning and return the model with the best number of iterations (trees).
            if (iteration % earlyStoppingRounds == 0)
            {
                var model = new ClassificationGradientBoostModel(
                    trees.Select(t => t.Take(iteration).ToArray()).ToArray(),
                    uniqueTargets, m_learningRate, initialLoss, trainingObservations.ColumnCount);

                var validPredictions = model.Predict(validationObservations);
                var error = metric.Error(validationTargets, validPredictions);

                Trace.WriteLine("Iteration " + (iteration + 1) + " Validation Error: " + error);

                if (currentBedstError >= error)
                {
                    currentBedstError = error;
                    bestIterationCount = iteration;
                }
            }
        }

        return new ClassificationGradientBoostModel(
            trees.Select(t => t.Take(bestIterationCount).ToArray()).ToArray(),
            uniqueTargets, m_learningRate, initialLoss, trainingObservations.ColumnCount);
    }

    /// <summary>
    /// Learns a ClassificationGradientBoostModel with early stopping.
    /// The parameter earlyStoppingRounds controls how often the validation error is measured.
    /// If the validation error has increased, the learning is stopped and the model with the best number of iterations (trees) is returned.
    /// The number of iterations used is equal to the number of trees in the resulting model.
    /// The method used for early stopping is based on the article:
    /// http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
    /// </summary>
    /// <param name="trainingObservations"></param>
    /// <param name="trainingTargets"></param>
    /// <param name="validationObservations"></param>
    /// <param name="validationTargets"></param>
    /// <param name="metric">The metric to use for early stopping</param>
    /// <param name="earlyStoppingRounds">This controls how often the validation error is checked to estimate the best number of iterations</param>
    /// <returns>ClassificationGradientBoostModel with early stopping. The number of iterations will equal the number of trees in the model</returns>
    public ClassificationGradientBoostModel LearnWithEarlyStopping(
        F64Matrix trainingObservations,
        double[] trainingTargets,
        F64Matrix validationObservations,
        double[] validationTargets,
        IMetric<double, ProbabilityPrediction> metric,
        int earlyStoppingRounds)
    {
        if (earlyStoppingRounds >= m_iterations)
        {
            throw new ArgumentException("Number of iterations " + m_iterations +
                " is smaller than earlyStoppingRounds " + earlyStoppingRounds);
        }

        var rows = trainingObservations.RowCount;
        var orderedElements = CreateOrderedElements(trainingObservations, rows);

        var inSample = trainingTargets.Select(t => false).ToArray();
        var indices = Enumerable.Range(0, trainingTargets.Length).ToArray();
        indices.ForEach(i => inSample[i] = true);
        var workIndices = indices.ToArray();

        var uniqueTargets = trainingTargets.Distinct().OrderBy(v => v).ToArray();
        var initialLoss = m_loss.InitialLoss(trainingTargets, inSample);

        double[][] oneVsAllTargets = null;
        double[][] predictions = null;
        double[][] residuals = null;
        GBMTree[][] trees = null;

        if (uniqueTargets.Length == 2) // Binary case - only need to fit to one class and use (1.0 - probability)
        {
            trees = new GBMTree[][] { new GBMTree[m_iterations] };
            predictions = new double[][] { trainingTargets.Select(_ => initialLoss).ToArray() };
            residuals = new double[][] { new double[trainingTargets.Length] };

            oneVsAllTargets = new double[1][];
            var target = uniqueTargets[0];
            oneVsAllTargets[0] = trainingTargets.Select(t => t == target ? 1.0 : 0.0).ToArray();

        }
        else // multi-class case - use oneVsAll strategy and fit probability for each class
        {
            trees = new GBMTree[uniqueTargets.Length][];
            predictions = uniqueTargets.Select(_ => trainingTargets.Select(t => initialLoss).ToArray())
                .ToArray();
            residuals = uniqueTargets.Select(_ => new double[trainingTargets.Length])
                .ToArray();

            oneVsAllTargets = new double[uniqueTargets.Length][];
            for (var i = 0; i < uniqueTargets.Length; i++)
            {
                var target = uniqueTargets[i];
                oneVsAllTargets[i] = trainingTargets.Select(t => t == target ? 1.0 : 0.0).ToArray();
                trees[i] = new GBMTree[m_iterations];
            }
        }

        var bestIterationCount = 0;
        var currentBedstError = double.MaxValue;

        for (var iteration = 0; iteration < m_iterations; iteration++)
        {
            for (var itarget = 0; itarget < trees.Length; itarget++)
            {
                m_loss.UpdateResiduals(oneVsAllTargets[itarget], predictions[itarget],
                    residuals[itarget], inSample);

                var sampleSize = trainingTargets.Length;
                if (m_subSampleRatio != 1.0)
                {
                    sampleSize = (int)Math.Round(m_subSampleRatio * workIndices.Length);
                    var currentInSample = Sample(sampleSize, workIndices, trainingTargets.Length);

                    trees[itarget][iteration] = m_learner.Learn(trainingObservations, oneVsAllTargets[itarget],
                        residuals[itarget], predictions[itarget], orderedElements, currentInSample);
                }
                else
                {
                    trees[itarget][iteration] = m_learner.Learn(trainingObservations, oneVsAllTargets[itarget],
                        residuals[itarget], predictions[itarget], orderedElements, inSample);
                }

                var predict = trees[itarget][iteration].Predict(trainingObservations);
                for (var i = 0; i < predict.Length; i++)
                {
                    predictions[itarget][i] += m_learningRate * predict[i];
                }
            }

            // When using early stopping, Check that the validation error is not increasing between earlyStoppingRounds
            // If the validation error has increased, stop the learning and return the model with the best number of iterations (trees).
            if (iteration % earlyStoppingRounds == 0)
            {
                var model = new ClassificationGradientBoostModel(
                    trees.Select(t => t.Take(iteration).ToArray()).ToArray(),
                    uniqueTargets, m_learningRate, initialLoss, trainingObservations.ColumnCount);

                var validPredictions = model.PredictProbability(validationObservations);
                var error = metric.Error(validationTargets, validPredictions);

                Trace.WriteLine("Iteration " + (iteration + 1) + " Validation Error: " + error);

                if (currentBedstError >= error)
                {
                    currentBedstError = error;
                    bestIterationCount = iteration;
                }
            }
        }

        return new ClassificationGradientBoostModel(
            trees.Select(t => t.Take(bestIterationCount).ToArray()).ToArray(),
            uniqueTargets, m_learningRate, initialLoss, trainingObservations.ColumnCount);
    }

    /// <summary>
    /// Private explicit interface implementation for indexed learning.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    IPredictorModel<double> IIndexedLearner<double>.Learn(
        F64Matrix observations, double[] targets, int[] indices) => Learn(observations, targets, indices);

    /// <summary>
    /// Private explicit interface implementation for indexed probability learning.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    IPredictorModel<ProbabilityPrediction> IIndexedLearner<ProbabilityPrediction>.Learn(
        F64Matrix observations, double[] targets, int[] indices) => Learn(observations, targets, indices);

    /// <summary>
    /// Private explicit interface implementation.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    IPredictorModel<double> ILearner<double>.Learn(
        F64Matrix observations, double[] targets) => Learn(observations, targets);

    /// <summary>
    /// Private explicit interface implementation for probability learning.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    IPredictorModel<ProbabilityPrediction> ILearner<ProbabilityPrediction>.Learn(
        F64Matrix observations, double[] targets) => Learn(observations, targets);

    /// <summary>
    /// Creates a matrix of ordered indices. Each row is ordered after the corresponding feature column.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="rows"></param>
    /// <returns></returns>
    static int[][] CreateOrderedElements(F64Matrix observations, int rows)
    {
        var orderedElements = new int[observations.ColumnCount][];

        for (var i = 0; i < observations.ColumnCount; i++)
        {
            var feature = observations.Column(i);
            var indices = Enumerable.Range(0, rows).ToArray();
            feature.SortWith(indices);
            orderedElements[i] = indices;
        }
        return orderedElements;
    }

    /// <summary>
    /// Creates a bool array with the selected samples (true)
    /// </summary>
    /// <param name="sampleSize"></param>
    /// <param name="indices"></param>
    /// <param name="allObservationCount"></param>
    /// <returns></returns>
    bool[] Sample(int sampleSize, int[] indices, int allObservationCount)
    {
        var inSample = new bool[allObservationCount];
        indices.Shuffle(m_random);

        for (var i = 0; i < sampleSize; i++)
        {
            inSample[indices[i]] = true;
        }

        return inSample;
    }
}
