using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using System;

namespace SharpLearning.CrossValidation.TrainingValidationSplitters
{
    /// <summary>
    /// Extension methods for ITrainingValidationIndexSplitters
    /// </summary>
    public static class TrainingValidationIndexSplitterExtensions
    {
        /// <summary>
        /// Splits the observations and targets into a training and a test set.
        /// </summary>
        /// <param name="splitter">The type of splitter used for dertermining the distribution of observations</param>
        /// <param name="observations">The observations for the problem</param>
        /// <param name="targets">The targets for the problem</param>
        /// <returns></returns>
        public static TrainingTestSetSplit SplitSet(this ITrainingValidationIndexSplitter<double> splitter, 
            F64Matrix observations, double[] targets)
        {
            if (observations.GetNumberOfRows() != targets.Length)
            { throw new ArgumentException("Observations and targets has different number of rows"); }

            var indexSplit = splitter.Split(targets);
            var trainingSet = new ObservationTargetSet((F64Matrix)observations.GetRows(indexSplit.TrainingIndices),
                targets.GetIndices(indexSplit.TrainingIndices));

            var testSet = new ObservationTargetSet((F64Matrix)observations.GetRows(indexSplit.ValidationIndices),
                targets.GetIndices(indexSplit.ValidationIndices));

            return new TrainingTestSetSplit(trainingSet, testSet);
        }
    }
}
