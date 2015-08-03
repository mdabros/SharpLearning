using System;
using System.Linq;

namespace SharpLearning.CrossValidation.TrainingTestSplitters
{
    /// <summary>
    /// Container for training and test indices
    /// </summary>
    public sealed class TrainingTestIndexSplit : IEquatable<TrainingTestIndexSplit>
    {
        public readonly int[] TrainingIndices;
        public readonly int[] TestIndices;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="trainingIndices"></param>
        /// <param name="testIndices"></param>
        public TrainingTestIndexSplit(int[] trainingIndices, int[] testIndices)
        {
            if (trainingIndices == null) { throw new ArgumentNullException("trainingIndices"); }
            if (testIndices == null) { throw new ArgumentNullException("validationIndices"); }
            TrainingIndices = trainingIndices;
            TestIndices = testIndices;
        }

        public bool Equals(TrainingTestIndexSplit other)
        {
            if (!this.TrainingIndices.SequenceEqual(other.TrainingIndices)) { return false; }
            if (!this.TestIndices.SequenceEqual(other.TestIndices)) { return false; }

            return true;
        }

        public override bool Equals(object obj)
        {
            TrainingTestIndexSplit other = obj as TrainingTestIndexSplit;
            if (other != null && Equals(other))
            {
                return true;
            }

            return false;
        }

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
