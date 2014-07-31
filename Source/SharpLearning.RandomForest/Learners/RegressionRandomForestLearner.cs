using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.RandomForest.Models;
using SharpLearning.Threading;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.RandomForest.Learners
{
    /// <summary>
    /// Trains a regression random forest
    /// http://en.wikipedia.org/wiki/Random_forest
    /// http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    /// </summary>
    public sealed class RegressionRandomForestLearner
    {
        readonly int m_trees;
        int m_featuresPrSplit;
        readonly int m_minimumSplitSize;
        readonly double m_minimumInformationGain;
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
        /// <param name="featuresPrSplit">Number of features used at each split in each tree. 0 means (availible features) / 3.0</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="seed">Seed for the random number generator</param>
        /// <param name="numberOfThreads">Number of threads to use for paralization</param>
        public RegressionRandomForestLearner(int trees, int minimumSplitSize, int maximumTreeDepth,
            int featuresPrSplit, double minimumInformationGain, int seed, int numberOfThreads)
        {
            if (trees < 1) { throw new ArgumentException("trees must be at least 1"); }
            if (featuresPrSplit < 0) { throw new ArgumentException("features pr split must be at least 1"); }
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            if (maximumTreeDepth <= 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
            if (numberOfThreads < 1) { throw new ArgumentException("Number of threads must be at least 1"); }

            m_trees = trees;
            m_minimumSplitSize = minimumSplitSize;
            m_maximumTreeDepth = maximumTreeDepth;
            m_featuresPrSplit = featuresPrSplit;
            m_minimumInformationGain = minimumInformationGain;
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
        /// <param name="seed">Seed for the random number generator</param>
        public RegressionRandomForestLearner(int trees = 100, int minimumSplitSize = 1, int maximumTreeDepth = 2000, 
            int featuresPrSplit = 0, double minimumInformationGain = .000001, int seed = 42)
          : this(trees, minimumSplitSize, maximumTreeDepth, featuresPrSplit, minimumInformationGain, 
            seed, Environment.ProcessorCount)
        {
        }

        /// <summary>
        /// Learns a classification random forest
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public RegressionForestModel Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Learns a classification random forest
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public RegressionForestModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            if (m_featuresPrSplit == 0)
            {
                var count = (int)(observations.GetNumberOfColumns() / 3.0);
                m_featuresPrSplit = count <= 0 ? 1 : count;
            }

            var results = new ConcurrentBag<RegressionDecisionTreeModel>();
            
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
            var rawVariableImportance = VariableImportance(models, observations.GetNumberOfColumns());

            return new RegressionForestModel(models, rawVariableImportance);
        }

        double[] VariableImportance(RegressionDecisionTreeModel[] models, int numberOfFeatures)
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
            ConcurrentBag<RegressionDecisionTreeModel> models, ConcurrentQueue<int> workItems)
        {
            var learner = new RegressionDecisionTreeLearner(m_maximumTreeDepth, m_minimumSplitSize, m_featuresPrSplit,
                m_minimumInformationGain, random.Next());

            var treeIndices = new int[indices.Length];

            int task = -1;
            while (workItems.TryDequeue(out task))
            {
                for (int j = 0; j < indices.Length; j++)
                {
                    treeIndices[j] = indices[random.Next(treeIndices.Length)];
                }

                var model = learner.Learn(observations, targets, treeIndices);
                models.Add(model);
            }
        }
    }
}
