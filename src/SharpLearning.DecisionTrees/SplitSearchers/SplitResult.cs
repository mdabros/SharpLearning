using System;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    /// <summary>
    /// 
    /// </summary>
    public struct SplitResult : IEquatable<SplitResult>
    {
        /// <summary>
        /// Split index within the feature used for split
        /// </summary>
        public readonly int SplitIndex;

        /// <summary>
        /// Threshold used for splitting
        /// </summary>
        public readonly double Threshold;

        /// <summary>
        /// Impurity imporvement obtained by making the split
        /// </summary>
        public readonly double ImpurityImprovement;

        /// <summary>
        /// Impurity of the left side of the split 
        /// </summary>
        public readonly double ImpurityLeft;

        /// <summary>
        /// Impurity of the right side of the split 
        /// </summary>
        public readonly double ImpurityRight;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="splitIndex">Split index within the feature used for split</param>
        /// <param name="threshold">Threshold used for splitting</param>
        /// <param name="impurityImprovement">Impurity imporvement obtained by making the split</param>
        /// <param name="impurityLeft">Impurity of the left side of the split </param>
        /// <param name="impurityRight">Impurity of the right side of the split</param>
        public SplitResult(int splitIndex, double threshold, double impurityImprovement,
            double impurityLeft, double impurityRight)
        {
            SplitIndex = splitIndex;
            Threshold = threshold;
            ImpurityImprovement = impurityImprovement;
            ImpurityLeft = impurityLeft;
            ImpurityRight = impurityRight;
        }

        /// <summary>
        /// Returns an initial SplitResult with start values
        /// </summary>
        /// <returns></returns>
        public static SplitResult Initial()
        {
            return new SplitResult(-1, 0.0, 0.0, 0.0, 0.0);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(SplitResult other)
        {
            if (SplitIndex != other.SplitIndex) { return false; }
            if (!Equal(Threshold, other.Threshold)) { return false; }
            if (!Equal(ImpurityImprovement, other.ImpurityImprovement)) { return false; }
            if (!Equal(ImpurityLeft, other.ImpurityLeft)) { return false; }
            if (!Equal(ImpurityRight, other.ImpurityRight)) { return false; }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (obj is SplitResult)
                return Equals((SplitResult)obj);
            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator ==(SplitResult p1, SplitResult p2)
        {
            return p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator !=(SplitResult p1, SplitResult p2)
        {
            return !p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return SplitIndex.GetHashCode() ^
                Threshold.GetHashCode() ^
                ImpurityImprovement.GetHashCode() ^
                ImpurityLeft.GetHashCode() ^
                ImpurityRight.GetHashCode();
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
