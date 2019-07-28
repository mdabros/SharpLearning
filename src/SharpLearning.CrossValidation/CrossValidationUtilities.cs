using System.Collections.Generic;
using System.Linq;
using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation
{
    /// <summary>
    /// Utilities for CrossValidation.
    /// </summary>
    public static class CrossValidationUtilities
    {
        /// <summary>
        /// Returns a list of (trainingIndices, validationIndices) 
        /// for use with k-fold cross-validation.
        /// </summary>
        /// <param name="sampler"></param>
        /// <param name="foldCount"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public static List<(int[] trainingIndices, int[] validationIndices)> GetCrossValidationIndexSets(
            IIndexSampler<double> sampler, int foldCount, double[] targets)
        {
            var allIndices = Enumerable.Range(0, targets.Length).ToArray();
            return GetCrossValidationIndexSets(sampler, foldCount, targets, allIndices);
        }

        /// <summary>
        /// Returns a list of (trainingIndices, validationIndices) 
        /// for use with k-fold cross-validation.
        /// </summary>
        /// <param name="sampler"></param>
        /// <param name="foldCount"></param>
        /// <param name="targets"></param>
        /// <param name="indices">indices to use for the cross validation sets</param>
        /// <returns></returns>
        public static List<(int[] trainingIndices, int[] validationIndices)> GetCrossValidationIndexSets(
            IIndexSampler<double> sampler, int foldCount, double[] targets, int[] indices)
        {
            var samplesPerFold = targets.Length / foldCount;
            var currentIndices = indices.ToArray();

            var crossValidationIndexSets = new List<(int[] training, int[] validation)>();

            for (int i = 0; i < foldCount; i++)
            {
                var holdoutSample = sampler.Sample(targets, samplesPerFold, currentIndices);
                // Sample only from remaining indices.
                currentIndices = currentIndices.Except(holdoutSample).ToArray();
                // Training sample is all indices except the current hold out sample.
                var trainingSample = indices.Except(holdoutSample).ToArray();
                crossValidationIndexSets.Add((trainingSample, holdoutSample));
            }

            return crossValidationIndexSets;
        }
    }
}
