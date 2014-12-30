using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.CrossValidation.BiasVarianceAnalysis
{
    /// <summary>
    /// Container for bias variance analysis data point
    /// </summary>
    public struct BiasVarianceLearningCurvePoint : IEquatable<BiasVarianceLearningCurvePoint>
    {
        public readonly int SampleSize;
        public readonly double TrainingScore;
        public readonly double ValidationScore;

        /// <summary>
        /// Container for bias variance analysis data point
        /// </summary>
        /// <param name="sampleSize"></param>
        /// <param name="trainingScore"></param>
        /// <param name="validationScore"></param>
        public BiasVarianceLearningCurvePoint(int sampleSize, double trainingScore, double validationScore)
        {
            SampleSize = sampleSize;
            TrainingScore = trainingScore;
            ValidationScore = validationScore;
        }

        public bool Equals(BiasVarianceLearningCurvePoint other)
        {
            if (this.SampleSize != other.SampleSize) { return false; }
            if (this.TrainingScore != other.TrainingScore) { return false; }
            if (this.ValidationScore != other.ValidationScore) { return false; }

            return true;
        }

        public static bool operator !=(BiasVarianceLearningCurvePoint x, BiasVarianceLearningCurvePoint y)
        {
            return !(x == y);
        }

        public static bool operator ==(BiasVarianceLearningCurvePoint x, BiasVarianceLearningCurvePoint y)
        {
            return (x.SampleSize == y.SampleSize) &&
                   (x.TrainingScore == y.TrainingScore) &&
                   (x.ValidationScore == y.ValidationScore);
        }

        public override bool Equals(object obj)
        {
            if (obj is BiasVarianceLearningCurvePoint)
                return this.Equals((BiasVarianceLearningCurvePoint)obj);
            else
                return false;
        }

        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                int hash = 17;
                hash = hash * 23 + SampleSize.GetHashCode();
                hash = hash * 23 + TrainingScore.GetHashCode();
                hash = hash * 23 + ValidationScore.GetHashCode();

                return hash;
            }
        }
    }
}
