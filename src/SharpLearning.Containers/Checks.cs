using System;
using System.Linq;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;

namespace SharpLearning.Containers
{
    /// <summary>
    /// Class containing common argument checks for the learners.
    /// </summary>
    public static class Checks
    {
        /// <summary>
        /// Verify that observations and targets are valid.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        public static void VerifyObservationsAndTargets(F64MatrixView observations, double[] targets)
        {
            VerifyObservationsAndTargets(observations.RowCount, observations.ColumnCount, targets.Length);
        }

        /// <summary>
        /// Verify that observations and targets are valid.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        public static void VerifyObservationsAndTargets(F64Matrix observations, double[] targets)
        {
            VerifyObservationsAndTargets(observations.RowCount, observations.ColumnCount, targets.Length);
        }

        /// <summary>
        /// Verify that observations and targets are valid.
        /// </summary>
        /// <param name="observationsRowCount"></param>
        /// <param name="observationsColumnCount"></param>
        /// <param name="targetLength"></param>
        public static void VerifyObservationsAndTargets(int observationsRowCount, 
            int observationsColumnCount, int targetLength)
        {
            VerifyObservations(observationsRowCount, observationsColumnCount);
            VerifyTargets(targetLength);
            VerifyObservationsAndTargetsDimensions(observationsRowCount, targetLength);
        }

        /// <summary>
        /// Verify that the observation matrix is valid.
        /// </summary>
        /// <param name="rowCount"></param>
        /// <param name="columnCount"></param>
        public static void VerifyObservations(int rowCount, int columnCount)
        {
            if(rowCount == 0)
            {
                throw new ArgumentException("Observations does not contain any rows");
            }

            if (columnCount == 0)
            {
                throw new ArgumentException("Observations does not contain any columns");
            }
        }

        /// <summary>
        /// Verify that the target vector is valid.
        /// </summary>
        /// <param name="targetLength"></param>
        public static void VerifyTargets(int targetLength)
        {
            if (targetLength == 0)
            {
                throw new ArgumentException("Targets does not contain any rows");
            }
        }

        /// <summary>
        /// Verify that observations and targets dimensions match.
        /// </summary>
        /// <param name="observationRowCount"></param>
        /// <param name="targetLength"></param>
        public static void VerifyObservationsAndTargetsDimensions(int observationRowCount, int targetLength)
        {
            if(observationRowCount != targetLength)
            {
                throw new ArgumentException($"Observations and targets mismatch." + 
                    $"Observations row count: {observationRowCount}, targets row count: {targetLength}");
            }           
        }

        /// <summary>
        /// Verify that indices are valid and match observations and targets.
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        public static void VerifyIndices(int[] indices, F64MatrixView observations, double[] targets)
        {
            VerifyIndices(indices, observations.RowCount, targets.Length);
        }

        /// <summary>
        /// Verify that indices are valid and match observations and targets.
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        public static void VerifyIndices(int[] indices, F64Matrix observations, double[] targets)
        {
            VerifyIndices(indices, observations.RowCount, targets.Length);
        }

        /// <summary>
        /// Verify that indices are valid and match observations and targets.
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="observationRowCount"></param>
        /// <param name="targetLength"></param>
        public static void VerifyIndices(int[] indices, int observationRowCount, int targetLength)
        {
            var min = indices.Min();
            if(min < 0)
            {
                throw new ArgumentException($"Indices contains negative " + 
                    $"values: {string.Join(",", indices.Where(v => v < 0))}");
            }

            var max = indices.Max();
            if (max >= observationRowCount || max >= targetLength)
            {
                throw new ArgumentException($"Indices contains elements exceeding the row count of observations and targets. " +  
                    $"Indices Max: {max}, observations row count: {observationRowCount}, target length: {targetLength}");
            }
        }

        /// <summary>
        /// Verify that featuresToUse is smaller or equal to featureCount.
        /// </summary>
        /// <param name="featuresToUse"></param>
        /// <param name="featureCount"></param>
        public static void VerifyFeaturesToUse(int featuresToUse, int featureCount)
        {
            if (featuresToUse > featureCount)
            {
                throw new InvalidOperationException(
                    $"Trying to use {featuresToUse} features per split, but only {featureCount} are available.");
            }
        }
    }
}
