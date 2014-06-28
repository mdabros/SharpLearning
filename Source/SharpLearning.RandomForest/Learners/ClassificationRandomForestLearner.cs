using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.FeatureCandidateSelectors;
using SharpLearning.DecisionTrees.LeafFactories;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Metrics.Entropy;
using SharpLearning.RandomForest.Models;
using System;
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
        readonly Random m_random;
        int[] m_workIndices = new int[0];

        /// <summary>
        /// The random forest is an ensemble learner consisting of a series of randomized decision trees
        /// </summary>
        /// <param name="trees">Number of trees to use in the ensemble</param>
        /// <param name="featuresPrSplit">Number of features used at each split in each tree</param>
        /// <param name="seed">Seed for the random number generator</param>
        public ClassificationRandomForestLearner(int trees, int featuresPrSplit, int seed)
        {
            if (trees < 1) { throw new ArgumentException("trees must be at least 1"); }
            if (featuresPrSplit < 1) { throw new ArgumentException("features pr split must be at least 1"); }
            m_trees = trees;
            m_featuresPrSplit = featuresPrSplit;

            m_random = new Random(seed);
        }

        /// <summary>
        /// Learns a classification random forest
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public ClassificationRandomForestModel Learn(F64Matrix observations, double[] targets)
        {
            Array.Resize(ref m_workIndices, targets.Length);
            var models = new ClassificationCartModel[m_trees];
            var treeIndices = new int[m_workIndices.Length];
            var rawVariableImportance = new double[observations.GetNumberOfColumns()];
            
            for (int i = 0; i < m_trees; i++)
            {
                for (int j = 0; j < treeIndices.Length; j++)
                {
                    treeIndices[j] = m_random.Next(treeIndices.Length);
                }

                var distinct = treeIndices.Distinct().Count();
                var model = CreateTreeModel(observations, targets, treeIndices);
                var modelVariableImportance = model.GetRawVariableImportance();
                
                for (int j = 0; j < modelVariableImportance.Length; j++)
                {
                    rawVariableImportance[j] += modelVariableImportance[j];
                }

                models[i] = model;
            }

            return new ClassificationRandomForestModel(models, rawVariableImportance);
        }

        ClassificationCartModel CreateTreeModel(F64Matrix observations, double[] targets, int[] indices)
        {
            var learner = new CartLearner(5, 100, m_featuresPrSplit, 0.0001, new GiniImpurityMetric(),
                new RandomFeatureCandidateSelector(m_random.Next()),
                new ClassificationLeafFactory());

            return new ClassificationCartModel(learner.Learn(observations, targets, indices),
                learner.m_variableImportance);
        }
    }
}
