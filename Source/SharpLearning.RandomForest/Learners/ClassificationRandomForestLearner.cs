using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.FeatureCandidateSelectors;
using SharpLearning.DecisionTrees.LeafFactories;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.Metrics.Entropy;
using SharpLearning.RandomForest.Models;
using SharpLearning.Threading;
using System;
using System.Collections.Concurrent;
using System.Linq;

namespace SharpLearning.RandomForest.Learners
{
    /// <summary>
    /// Trains a classification random forest
    /// http://en.wikipedia.org/wiki/Random_forest
    /// http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    /// </summary>
    public sealed class ClassificationRandomForestLearner
    {
        readonly int m_trees;
        readonly int m_featuresPrSplit;
        readonly int m_minimumSplitSize;
        readonly double m_minimumInformationGain;
        readonly int m_maximumTreeDepth;
        readonly Random m_random;

        readonly ThreadedWorker<ClassificationDecisionTreeModel> m_threadedWorker;

        /// <summary>
        /// The random forest is an ensemble learner consisting of a series of randomized decision trees
        /// </summary>
        /// <param name="trees">Number of trees to use in the ensemble</param>
        /// <param name="minimumSplitSize">The minimum size for a node to be split</param>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="featuresPrSplit">Number of features used at each split in each tree</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="seed">Seed for the random number generator</param>
        /// <param name="numberOfThreads">Number of threads to use for paralization</param>
        public ClassificationRandomForestLearner(int trees, int minimumSplitSize, int maximumTreeDepth, 
            int featuresPrSplit, double minimumInformationGain, int seed, int numberOfThreads)
        {
            if (trees < 1) { throw new ArgumentException("trees must be at least 1"); }
            if (featuresPrSplit < 1) { throw new ArgumentException("features pr split must be at least 1"); }
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            if (maximumTreeDepth <= 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
            if (numberOfThreads < 1) { throw new ArgumentException("Number of threads must be at least 1"); }

            m_trees = trees;
            m_minimumSplitSize = minimumSplitSize;
            m_maximumTreeDepth = maximumTreeDepth;
            m_featuresPrSplit = featuresPrSplit;
            m_minimumInformationGain = minimumInformationGain;

            m_random = new Random(seed);

            m_threadedWorker = new ThreadedWorker<ClassificationDecisionTreeModel>(numberOfThreads);
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
        public ClassificationRandomForestLearner(int trees = 50, int minimumSplitSize = 5, int maximumTreeDepth = 100, 
            int featuresPrSplit = 3, double minimumInformationGain = 0.0001, int seed = 42)
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
        public ClassificationRandomForestModel Learn(F64Matrix observations, double[] targets)
        {          
            var results = new ConcurrentBag<ClassificationDecisionTreeModel>();
            var tasks = new ConcurrentQueue<Action<ConcurrentBag<ClassificationDecisionTreeModel>>>();
            
            for (int i = 0; i < m_trees; i++)
            {
                tasks.Enqueue((r) => CreateTreeModel(observations, targets, 
                    new Random(m_random.Next()), r));
            }

            m_threadedWorker.Run(tasks, results);

            var models = results.ToArray();
            var rawVariableImportance = VariableImportance(models, observations.GetNumberOfColumns());

            return new ClassificationRandomForestModel(models, rawVariableImportance);
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

        void CreateTreeModel(F64Matrix observations, double[] targets, Random random, 
            ConcurrentBag<ClassificationDecisionTreeModel> models)
        {
            var treeIndices = new int[targets.Length];

            for (int j = 0; j < treeIndices.Length; j++)
            {
                treeIndices[j] = random.Next(treeIndices.Length);
            }
            
            var learner = new DecisionTreeLearner(m_maximumTreeDepth, 
                m_featuresPrSplit, m_minimumInformationGain,
                new GiniImpurityMetric(),
                new LinearSplitSearcher(m_minimumSplitSize),
                new RandomFeatureCandidateSelector(random.Next()),
                new ClassificationLeafFactory());

            var model = new ClassificationDecisionTreeModel(learner.Learn(observations, targets, treeIndices),
                learner.m_variableImportance);

            models.Add(model);
        }
    }
}
