using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Containers
{
    /// <summary>
    /// Probability prediction for classification learners with probability estimates
    /// </summary>
    [Serializable]
    public struct ProbabilityPrediction : IEquatable<ProbabilityPrediction>
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly double Prediction;

        /// <summary>
        /// 
        /// </summary>
        public readonly Dictionary<double, double> Probabilities;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="prediction"></param>
        /// <param name="probabilities">Dictionary containing the class name to class probability</param>
        public ProbabilityPrediction(double prediction, Dictionary<double, double> probabilities)
        {
            Probabilities = probabilities ?? throw new ArgumentNullException(nameof(probabilities));
            Prediction = prediction;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(ProbabilityPrediction other)
        {
            if(!Equal(this.Prediction, other.Prediction)) { return false; }
            if (this.Probabilities.Count != other.Probabilities.Count) { return false; }
            
            var zip = this.Probabilities.Zip(other.Probabilities, (t, o) => new {This = t, Other = o});
            foreach (var item in zip)
            {
                if (item.This.Key != item.Other.Key)
                    return false;
                if (!Equal(item.This.Value, item.Other.Value))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if(obj is ProbabilityPrediction)
                return Equals((ProbabilityPrediction)obj);
            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator ==(ProbabilityPrediction p1, ProbabilityPrediction p2)
        {
            return p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator !=(ProbabilityPrediction p1, ProbabilityPrediction p2)
        {
            return !p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return Prediction.GetHashCode() ^ Probabilities.GetHashCode();
        }

        const double m_tolerence = 0.00001;

        bool Equal(double a, double b)
        {
            var diff = Math.Abs(a * m_tolerence);
            if(Math.Abs(a - b) <= diff)
            {
                return true;
            }

            return false;
        }
    }
}
