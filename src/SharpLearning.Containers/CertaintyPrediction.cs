using System;

namespace SharpLearning.Containers
{
    /// <summary>
    /// Certainty prediction for regression learners with certainty estimate
    /// </summary>
    [Serializable]
    public struct CertaintyPrediction
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly double Prediction;

        /// <summary>
        /// 
        /// </summary>
        public readonly double Variance;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="prediction"></param>
        /// <param name="variance"></param>
        public CertaintyPrediction(double prediction, double variance)
        {
            Variance = variance;
            Prediction = prediction;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(CertaintyPrediction other)
        {
            if (!Equal(this.Prediction, other.Prediction)) { return false; }
            if (!Equal(this.Variance, other.Variance)) { return false; }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (obj is CertaintyPrediction)
                return Equals((CertaintyPrediction)obj);
            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator ==(CertaintyPrediction p1, CertaintyPrediction p2)
        {
            return p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator !=(CertaintyPrediction p1, CertaintyPrediction p2)
        {
            return !p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return Prediction.GetHashCode() ^ Variance.GetHashCode();
        }

        const double m_tolerence = 0.00001;

        bool Equal(double a, double b)
        {
            var diff = Math.Abs(a * m_tolerence);
            if (Math.Abs(a - b) <= diff)
            {
                return true;
            }

            return false;
        }
    }
}
