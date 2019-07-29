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
        public static IReadOnlyList<(int[] trainingIndices, int[] validationIndices)> GetKFoldCrossValidationIndexSets<T>(
            IIndexSampler<T> sampler, int foldCount, T[] targets)
        {
            var allIndices = Enumerable.Range(0, targets.Length).ToArray();
            return GetKFoldCrossValidationIndexSets(sampler, foldCount, targets, allIndices);
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
        public static IReadOnlyList<(int[] trainingIndices, int[] validationIndices)> GetKFoldCrossValidationIndexSets<T>(
            IIndexSampler<T> sampler, int foldCount, T[] targets, int[] indices)
        {
            var samplesPerFold = indices.Length / foldCount;

            var currentIndices = indices.ToArray();
            var crossValidationIndexSets = new List<(int[] training, int[] validation)>();
            
            for (int i = 0; i < foldCount; i++)
            {
                var lastFold = (i == foldCount - 1);
                int[] validationIndices;

                if (lastFold) // handle remainders.
                {
                    // Last fold. Add remaining indices.
                    // This is done to ensure that all indices are included,
                    // even if the targets.Length % foldCount != 0.
                    // Note, that this will make the ratio between training and validation
                    // different for the last set compared to the others.
                    validationIndices = currentIndices.ToArray();
                }
                else
                {
                    validationIndices = sampler.Sample(targets, samplesPerFold, currentIndices);
                    // Sample only from remaining indices.
                    currentIndices = currentIndices.Except(validationIndices).ToArray();
                }

                // Training sample is all indices except the current validation sample.
                var trainingIndices = indices.Except(validationIndices).ToArray();
                crossValidationIndexSets.Add((trainingIndices, validationIndices));
            }

            return crossValidationIndexSets;
        }
    }
}
