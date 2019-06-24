using System;

namespace SharpLearning.CrossValidation.LearningCurves
{
    /// <summary>
    /// Container for learning curve data point
    /// </summary>
    public struct LearningCurvePoint : IEquatable<LearningCurvePoint>
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly int SampleSize;

        /// <summary>
        /// 
        /// </summary>
        public readonly double TrainingScore;

        /// <summary>
        /// 
        /// </summary>
        public readonly double ValidationScore;

        /// <summary>
        /// Container for learning curve data point
        /// </summary>
        /// <param name="sampleSize"></param>
        /// <param name="trainingScore"></param>
        /// <param name="validationScore"></param>
        public LearningCurvePoint(int sampleSize, double trainingScore, double validationScore)
        {
            SampleSize = sampleSize;
            TrainingScore = trainingScore;
            ValidationScore = validationScore;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(LearningCurvePoint other)
        {
            if (this.SampleSize != other.SampleSize) { return false; }
            if (this.TrainingScore != other.TrainingScore) { return false; }
            if (this.ValidationScore != other.ValidationScore) { return false; }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static bool operator !=(LearningCurvePoint x, LearningCurvePoint y)
        {
            return !(x == y);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static bool operator ==(LearningCurvePoint x, LearningCurvePoint y)
        {
            return (x.SampleSize == y.SampleSize) &&
                   (x.TrainingScore == y.TrainingScore) &&
                   (x.ValidationScore == y.ValidationScore);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (obj is LearningCurvePoint)
                return this.Equals((LearningCurvePoint)obj);
            else
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
                hash = hash * 23 + SampleSize.GetHashCode();
                hash = hash * 23 + TrainingScore.GetHashCode();
                hash = hash * 23 + ValidationScore.GetHashCode();

                return hash;
            }
        }
    }
}
