using System;
using System.Linq;

namespace SharpLearning.CrossValidation.TrainingValidationSplitters
{
    /// <summary>
    /// Container for training and validation indices
    /// </summary>
    public sealed class TrainingValidationIndexSplit : IEquatable<TrainingValidationIndexSplit>
    {
        public readonly int[] TrainingIndices;
        public readonly int[] ValidationIndices;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="trainingIndices"></param>
        /// <param name="validationIndices"></param>
        public TrainingValidationIndexSplit(int[] trainingIndices, int[] validationIndices)
        {
            if (trainingIndices == null) { throw new ArgumentNullException("trainingIndices"); }
            if (validationIndices == null) { throw new ArgumentNullException("validationIndices"); }
            TrainingIndices = trainingIndices;
            ValidationIndices = validationIndices;
        }

        public bool Equals(TrainingValidationIndexSplit other)
        {
            if (!this.TrainingIndices.SequenceEqual(other.TrainingIndices)) { return false; }
            if (!this.ValidationIndices.SequenceEqual(other.ValidationIndices)) { return false; }

            return true;
        }

        public override bool Equals(object obj)
        {
            TrainingValidationIndexSplit other = obj as TrainingValidationIndexSplit;
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
                hash = hash * 23 + ValidationIndices.GetHashCode();

                return hash;
            }
        }
    }
}
