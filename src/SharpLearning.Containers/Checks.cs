using System;
using System.Linq;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers
{
    /// <summary>
    /// Class containing common argument checks for the learners.
    /// </summary>
    public static class Checks
    {
        public static void VerifyAllLearnerInputs(F64Matrix observations, double[] targets, int[] indices)
        {
            VerifyObservations(observations);
            VerifyTargets(targets);
            VerifyObservationsAndTargetsDimensionMatch(observations, targets);
            VerifyIndices(indices, observations, targets);
        }

        /// <summary>
        /// Verify that the observation matrix is valid.
        /// </summary>
        /// <param name="observations"></param>
        public static void VerifyObservations(F64Matrix observations)
        {
            if(observations.RowCount == 0)
            {
                throw new ArgumentException("Observations does not contain any rows");
            }

            if (observations.ColumnCount== 0)
            {
                throw new ArgumentException("Observations does not contain any columns");
            }
        }

        /// <summary>
        /// Verify that the target vector is valid.
        /// </summary>
        /// <param name="targets"></param>
        public static void VerifyTargets(double[] targets)
        {
            if (targets.Length == 0)
            {
                throw new ArgumentException("Targets does not contain any rows");
            }
        }

        /// <summary>
        /// Verify that observations and targets match.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        public static void VerifyObservationsAndTargetsDimensionMatch(F64Matrix observations, double[] targets)
        {
            if(observations.RowCount != targets.Length)
            {
                throw new ArgumentException($"Observation row count: {observations.RowCount} and target length: {targets.Length} does not math");
            }           
        }

        /// <summary>
        /// Verify that indices match observations and targets.
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        public static void VerifyIndices(int[] indices, F64Matrix observations, double[] targets)
        {
            var min = indices.Min();
            if(min > 0)
            {
                throw new ArgumentException($"Indices contains elements below zero: {min}");
            }

            var max = indices.Max();
            if (max >= observations.ColumnCount || max >= targets.Length)
            {
                throw new ArgumentException($"Indices contains elements larger than the rows of observations. Indices Max: {max}, observations row count: {observations.RowCount}, target length: {targets.Length}");
            }
        }
    }
}
