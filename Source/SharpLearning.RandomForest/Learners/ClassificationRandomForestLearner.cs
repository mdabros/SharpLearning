using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Common.Interfaces;
using SharpLearning.RandomForest.Models;
using SharpLearning.Threading;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.RandomForest.Learners
{
    /// <summary>
    /// Trains a classification random forest
    /// http://en.wikipedia.org/wiki/Random_forest
    /// http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    /// </summary>
    public sealed class ClassificationRandomForestLearner : IIndexedLearner<double>, IIndexedLearner<ProbabilityPrediction>,
        ILearner<double>, ILearner<ProbabilityPrediction>
       
    {
        readonly int m_trees;
        int m_featuresPrSplit;
        readonly int m_minimumSplitSize;
        readonly double m_minimumInformationGain;
        readonly double m_subSampleRatio;
        readonly int m_maximumTreeDepth;
        readonly Random m_random;
        readonly int m_numberOfThreads;

        WorkerRunner m_threadedWorker;

        /// <summary>
        /// The random forest is an ensemble learner consisting of a series of randomized decision trees
        /// </summary>
        /// <param name="trees">Number of trees to use in the ensemble</param>
        /// <param name="minimumSplitSize">The minimum size for a node to be split</param>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="featuresPrSplit">Number of features used at each split in each tree. 0 means Sqrt(of availible features)</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="subSampleRatio">The ratio of observations sampled with replacement for each tree. 
        /// Default is 1.0 sampling the same count as the number of observations in the input. 
        /// If below 1.0 the algorithm changes to random patches</param>
        /// <param name="seed">Seed for the random number generator</param>
        /// <param name="numberOfThreads">Number of threads to use for paralization</param>
        public ClassificationRandomForestLearner(int trees, int minimumSplitSize, int maximumTreeDepth,
            int featuresPrSplit, double minimumInformationGain, double subSampleRatio, int seed, int numberOfThreads)
        {
            if (trees < 1) { throw new ArgumentException("trees must be at least 1"); }
            if (featuresPrSplit < 0) { throw new ArgumentException("features pr split must be at least 1"); }
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            if (maximumTreeDepth <= 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
            if (subSampleRatio <= 0.0 || subSampleRatio > 1.0) { throw new ArgumentException("subSampleRatio must be larger than 0.0 and at max 1.0"); }
            if (numberOfThreads < 1) { throw new ArgumentException("Number of threads must be at least 1"); }

            m_trees = trees;
            m_minimumSplitSize = minimumSplitSize;
            m_maximumTreeDepth = maximumTreeDepth;
            m_featuresPrSplit = featuresPrSplit;
            m_minimumInformationGain = minimumInformationGain;
            m_subSampleRatio = subSampleRatio;
            m_numberOfThreads = numberOfThreads;

            m_random = new Random(seed);
        }

        /// <summary>
        /// The random forest is an ensemble learner consisting of a series of randomized decision trees
        /// </summary>
        /// <param name="trees">Number of trees to use in the ensemble</param>
        /// <param name="minimumSplitSize">The minimum size for a node to be split</param>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="featuresPrSplit">Number of features used at each split in each tree</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="subSampleRatio">The ratio of observations sampled with replacement for each tree. 
        /// Default is 1.0 sampling the same count as the number of observations in the input. 
        /// If below 1.0 the algorithm changes to random patches</param>
        /// <param name="seed">Seed for the random number generator</param>
        public ClassificationRandomForestLearner(int trees = 100, int minimumSplitSize = 1, int maximumTreeDepth = 2000,
            int featuresPrSplit = 0, double minimumInformationGain = .000001, double subSampleRatio = 1.0, int seed = 42)
          : this(trees, minimumSplitSize, maximumTreeDepth, featuresPrSplit, minimumInformationGain,
            subSampleRatio, seed, Environment.ProcessorCount)
        {
        }

        /// <summary>
        /// Learns a classification random forest
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public ClassificationForestModel Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Learns a classification random forest
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public ClassificationForestModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            if (m_featuresPrSplit == 0)
            {
                var count = (int)Math.Sqrt(observations.ColumnCount());
                m_featuresPrSplit = count <= 0 ? 1 : count;
            }

            var results = new ConcurrentBag<ClassificationDecisionTreeModel>();
            
            var workItems = new ConcurrentQueue<int>();
            for (int i = 0; i < m_trees; i++)
            {
                workItems.Enqueue(0);
            }

            var workers = new List<Action>();
            for (int i = 0; i < m_numberOfThreads; i++)
            {
                workers.Add(() => CreateTreeModel(observations, targets, indices, new Random(m_random.Next()),
                    results, workItems));                    
            }

            m_threadedWorker = new WorkerRunner(workers);
            m_threadedWorker.Run();

            var models = results.ToArray();
            var rawVariableImportance = VariableImportance(models, observations.ColumnCount());

            return new ClassificationForestModel(models, rawVariableImportance);
        }

        /// <summary>
        /// Private explicit interface implementation for indexed learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Private explicit interface implementation for indexed probability learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<ProbabilityPrediction> IIndexedLearner<ProbabilityPrediction>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Private explicit interface implementation for indexed learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        /// <summary>
        /// Private explicit interface implementation for indexed probability learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<ProbabilityPrediction> ILearner<ProbabilityPrediction>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }
        
        double[] VariableImportance(ClassificationDecisionTreeModel[] models, int numberOfFeatures)
        {
            var rawVariableImportance = new double[numberOfFeatures];

            foreach (var model in models)
            {
                var modelVariableImportance = model.GetRawVariableImportance();

                for (int j = 0; j < modelVariableImportance.Length; j++)
                {
                    rawVariableImportance[j] += modelVariableImportance[j];
                }
            }
            return rawVariableImportance;
        }

        void CreateTreeModel(F64Matrix observations, double[] targets, int[] indices, Random random, 
            ConcurrentBag<ClassificationDecisionTreeModel> models, ConcurrentQueue<int> workItems)
        {
            var learner = new ClassificationDecisionTreeLearner(m_maximumTreeDepth, m_minimumSplitSize, m_featuresPrSplit,
                m_minimumInformationGain, random.Next());

            var treeIndicesLength = (int)Math.Round(m_subSampleRatio * (double)indices.Length);
            var treeIndices = new int[treeIndicesLength];

            int task = -1;
            while (workItems.TryDequeue(out task))
            {
                for (int j = 0; j < treeIndicesLength; j++)
                {
                    treeIndices[j] = indices[random.Next(indices.Length)];
                }

                var model = learner.Learn(observations, targets, treeIndices);
                models.Add(model);
            }
        }
    }
}
