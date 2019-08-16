using System;
using System.Linq;

namespace SharpLearning.CrossValidation.TrainingTestSplitters
{
    /// <summary>
    /// Container for training and test indices
    /// </summary>
    public sealed class TrainingTestIndexSplit : IEquatable<TrainingTestIndexSplit>
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly int[] TrainingIndices;

        /// <summary>
        /// 
        /// </summary>
        public readonly int[] TestIndices;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="trainingIndices"></param>
        /// <param name="testIndices"></param>
        public TrainingTestIndexSplit(int[] trainingIndices, int[] testIndices)
        {
            TrainingIndices = trainingIndices ?? throw new ArgumentNullException(nameof(trainingIndices));
            TestIndices = testIndices ?? throw new ArgumentNullException(nameof(testIndices));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(TrainingTestIndexSplit other)
        {
            if (!this.TrainingIndices.SequenceEqual(other.TrainingIndices)) { return false; }
            if (!this.TestIndices.SequenceEqual(other.TestIndices)) { return false; }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (obj is TrainingTestIndexSplit other && this.Equals(other))
            {
                return true;
            }

            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                int hash = 17;
                hash = hash * 23 + TrainingIndices.GetHashCode();
                hash = hash * 23 + TestIndices.GetHashCode();

                return hash;
            }
        }
    }
}
