using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.Metrics.Regression;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.GradientBoost.GBM
{
    /// <summary>
    /// <summary>
    /// Regression gradient boost learner based on 
    /// http://statweb.stanford.edu/~jhf/ftp/trebst.pdf
    /// A series of regression trees are fitted stage wise on the residuals of the previous stage.
    /// The resulting models are ensembled together using addition. Implementation based on:
    /// http://gradientboostedmodels.googlecode.com/files/report.pdf
    /// </summary>
    /// </summary>
    public class GBMGradientBoostRegressorLearner
    {
        readonly GBMDecisionTreeLearner m_learner;
        readonly double m_learningRate;
        readonly int m_iterations;
        readonly double m_subSampleRatio;
        readonly Random m_random = new Random(42);
        readonly IGBMLoss m_loss;

        /// <summary>
        /// <summary>
        ///  Least squares (LS) regression gradient boost learner. 
        ///  A series of regression trees are fitted stage wise on the residuals of the previous stage
        /// </summary>
        /// <param name="iterations">The number of iterations or stages</param>
        /// <param name="learningRate">How much each iteration should contribute with</param>
        /// <param name="maximumTreeDepth">The maximum depth of the tree models</param>
        /// <param name="maximumLeafCount">The maximum leaf count of the tree models</param>
        /// <param name="minimumSplitSize">minimum node split size in the trees 1 is default</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="subSampleRatio">ratio of observations sampled at each iteration. Default is 1.0. 
        /// If below 1.0 the algorithm changes to stochastic gradient boosting. 
        /// This reduces variance in the ensemble and can help ounter overfitting</param>
        /// <param name="loss">loss function used</param>
        /// <param name="numberOfThreads">Number of threads to use for paralization</param>
        public GBMGradientBoostRegressorLearner(int iterations, double learningRate, int maximumTreeDepth,
            int minimumSplitSize, double minimumInformationGain, double subSampleRatio, IGBMLoss loss, int numberOfThreads)
        {
            if (iterations < 1) { throw new ArgumentException("Iterations must be at least 1"); }
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            if (maximumTreeDepth < 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
            if (subSampleRatio <= 0.0 || subSampleRatio > 1.0) { throw new ArgumentException("subSampleRatio must be larger than 0.0 and at max 1.0"); }
            if (loss == null) { throw new ArgumentException("loss"); }

            m_iterations = iterations;
            m_learningRate = learningRate;
            m_subSampleRatio = subSampleRatio;
            m_loss = loss;
            m_learner = new GBMDecisionTreeLearner(maximumTreeDepth, minimumSplitSize, minimumInformationGain, m_loss, numberOfThreads);
        }

        /// <summary>
        /// <summary>
        ///  Least squares (LS) regression gradient boost learner. 
        ///  A series of regression trees are fitted stage wise on the residuals of the previous stage
        /// </summary>
        /// <param name="iterations">The number of iterations or stages</param>
        /// <param name="learningRate">How much each iteration should contribute with</param>
        /// <param name="maximumTreeDepth">The maximum depth of the tree models</param>
        /// <param name="maximumLeafCount">The maximum leaf count of the tree models</param>
        /// <param name="minimumSplitSize">minimum node split size in the trees 1 is default</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="subSampleRatio">ratio of observations sampled at each iteration. Default is 1.0. 
        /// If below 1.0 the algorithm changes to stochastic gradient boosting. 
        /// This reduces variance in the ensemble and can help ounter overfitting</param>
        public GBMGradientBoostRegressorLearner(int iterations = 100, double learningRate = 0.1, int maximumTreeDepth = 3,
            int minimumSplitSize = 1, double minimumInformationGain = 0.000001, double subSampleRatio = 1.0)
            : this(iterations, learningRate, maximumTreeDepth, minimumSplitSize, minimumInformationGain, 
                subSampleRatio, new GBMSquaredLoss(), Environment.ProcessorCount)
        {
        }

        /// <summary>
        ///  Least squares (LS) regression gradient boost learner. 
        ///  A series of regression trees are fitted stage wise on the residuals of the previous tree
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public GBMGradientBoostRegressorModel Learn(F64Matrix observations, double[] targets)
        {
            var orderedElements = new int[observations.GetNumberOfColumns()][];
            var rows = observations.GetNumberOfRows();

            for (int i = 0; i < observations.GetNumberOfColumns(); i++)
            {
                var feature = observations.GetColumn(i);
                var indices = Enumerable.Range(0, rows).ToArray();
                feature.SortWith(indices);
                orderedElements[i] = indices;
            }

            var trees = new GBMTree[m_iterations];
            trees[0] = FitInitial(targets);

            var predictions = trees[0].Predict(observations);
            var residuals = new double[targets.Length];
            var residuals2 = new double[targets.Length];

            var allIndices = Enumerable.Range(0, targets.Length).ToArray();

            var inSample = targets.Select(t => true).ToArray();
            
            for (int iteration = 1; iteration < m_iterations; iteration++)
            {
                for (int j = 0; j < targets.Length; j++)
                {
                    var residual = NegativeGradient(targets[j], predictions[j]);
                    var residual2 = residual * residual;

                    residuals[j] = residual;
                    residuals2[j] = residual2;
                }
                var s = residuals.Sum();
                var s2 = residuals2.Sum();

                var sampleSize = targets.Length;
                if(m_subSampleRatio != 1.0)
                {
                    sampleSize = (int)Math.Round(m_subSampleRatio * targets.Length);
                    inSample = Sample(sampleSize, allIndices);
                }

                var tree = m_learner.Learn(observations, residuals, orderedElements,
                    inSample, s, s2, sampleSize);

                trees[iteration] = tree;

                var predict = tree.Predict(observations);

                for (int i = 0; i < predict.Length; i++)
                {
                    predictions[i] += m_learningRate * predict[i];
                }
            }

            return new GBMGradientBoostRegressorModel(trees, m_learningRate);
        }

        /// <summary>
        /// The negative gradient (least square)
        /// </summary>
        /// <param name="target"></param>
        /// <param name="prediction"></param>
        /// <returns></returns>
        double NegativeGradient(double target, double prediction)
        {
            return target - prediction;
        }

        /// <summary>
        /// Creates a bool array with the selected samples (true)
        /// </summary>
        /// <param name="sampleSize"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        bool[] Sample(int sampleSize, int[] indices)
        {
            var inSample = new bool[indices.Length];
            indices.Shuffle(m_random);

            for (int i = 0; i < sampleSize; i++)
            {
                inSample[indices[i]] = true;
            }
            
            return inSample;
        }

        /// <summary>
        /// Fits the initial best constant (least squares)
        /// </summary>
        /// <param name="targets"></param>
        /// <returns></returns>
        GBMTree FitInitial(double[] targets)
        {
            var n = targets.Length;
            var s = targets.Sum();
            var s2 = targets.Select(t => t * t).Sum();

            var rootBestConstant = s / (double)n;
            var rootCost = s2 - s * s / (double)n;
            var root = new GBMNode()
            {
                FeatureIndex = -1,
                SplitValue = -1,
                LeftError = rootCost,
                RightError = rootCost,
                LeftConstant = rootBestConstant,
                RightConstant = rootBestConstant
            };

            var nodes = new List<GBMNode> { root };
            return new GBMTree(nodes);
        }
    }
}
