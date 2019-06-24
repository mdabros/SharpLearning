using System;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.CrossValidation.TrainingTestSplitters
{
    /// <summary>
    /// Extension methods for ITrainingTestIndexSplitters
    /// </summary>
    public static class TrainingTestIndexSplitterExtensions
    {
        /// <summary>
        /// Splits the observations and targets into a training and a test set.
        /// </summary>
        /// <param name="splitter">The type of splitter used for determining the distribution of observations</param>
        /// <param name="observations">The observations for the problem</param>
        /// <param name="targets">The targets for the problem</param>
        /// <returns></returns>
        public static TrainingTestSetSplit SplitSet(this ITrainingTestIndexSplitter<double> splitter, 
            F64Matrix observations, double[] targets)
        {
            if (observations.RowCount != targets.Length)
            { throw new ArgumentException("Observations and targets has different number of rows"); }

            var indexSplit = splitter.Split(targets);
            var trainingSet = new ObservationTargetSet((F64Matrix)observations.Rows(indexSplit.TrainingIndices),
                targets.GetIndices(indexSplit.TrainingIndices));

            var testSet = new ObservationTargetSet((F64Matrix)observations.Rows(indexSplit.TestIndices),
                targets.GetIndices(indexSplit.TestIndices));

            return new TrainingTestSetSplit(trainingSet, testSet);
        }
    }
}
