
using System;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    public struct SplitResult : IEquatable<SplitResult>
    {
        public readonly int SplitIndex;
        public readonly double Threshold;
        public readonly double ImpurityImprovement;
        public readonly double ImpurityLeft;
        public readonly double ImpurityRight;

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

        public bool Equals(SplitResult other)
        {
            if (SplitIndex != other.SplitIndex) { return false; }
            if (!Equal(Threshold, other.Threshold)) { return false; }
            if (!Equal(ImpurityImprovement, other.ImpurityImprovement)) { return false; }
            if (!Equal(ImpurityLeft, other.ImpurityLeft)) { return false; }
            if (!Equal(ImpurityRight, other.ImpurityRight)) { return false; }

            return true;
        }

        public override bool Equals(object obj)
        {
            if (obj is SplitResult)
                return Equals((SplitResult)obj);
            return false;
        }


        public static bool operator ==(SplitResult p1, SplitResult p2)
        {
            return p1.Equals(p2);
        }

        public static bool operator !=(SplitResult p1, SplitResult p2)
        {
            return !p1.Equals(p2);
        }

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

    /// <summary>
    /// Structure for holding split results
    /// </summary>
    public struct FindSplitResult : IEquatable<FindSplitResult>
    {
        const double m_tolerence = 0.00001;

        public readonly int BestSplitIndex;
        public readonly double BestInformationGain;
        public readonly FeatureSplit BestFeatureSplit;

        public readonly IntervalEntropy LeftIntervalEntropy;
        public readonly IntervalEntropy RightIntervalEntropy;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="newBestSplit"></param>
        /// <param name="bestSplitIndex"></param>
        /// <param name="bestInformationGain"></param>
        /// <param name="bestFeatureSplit"></param>
        /// <param name="leftIntervalEntropy"></param>
        /// <param name="rightIntervalEntropy"></param>
        public FindSplitResult(int bestSplitIndex, double bestInformationGain, FeatureSplit bestFeatureSplit,
            IntervalEntropy leftIntervalEntropy, IntervalEntropy rightIntervalEntropy)
        {
            BestSplitIndex = bestSplitIndex;
            BestInformationGain = bestInformationGain;
            BestFeatureSplit = bestFeatureSplit;
            LeftIntervalEntropy = leftIntervalEntropy;
            RightIntervalEntropy = rightIntervalEntropy;
        }

        /// <summary>
        /// Returns an initial FindSplitResult with start values
        /// </summary>
        /// <returns></returns>
        public static FindSplitResult Initial()
        {
            return new FindSplitResult(-1, 0.0, new FeatureSplit(),
                IntervalEntropy.Initial(), IntervalEntropy.Initial());
        }

        public bool Equals(FindSplitResult other)
        {
            if (BestSplitIndex != other.BestSplitIndex) { return false; }
            if (!Equal(BestInformationGain, other.BestInformationGain)) { return false; }
            if (!BestFeatureSplit.Equals(other.BestFeatureSplit)) { return false; }
            if (!LeftIntervalEntropy.Equals(other.LeftIntervalEntropy)) { return false; }
            if (!RightIntervalEntropy.Equals(other.RightIntervalEntropy)) { return false; }

            return true;
        }

        public override bool Equals(object obj)
        {
            if (obj is FindSplitResult)
                return Equals((FindSplitResult)obj);
            return false;
        }


        public static bool operator ==(FindSplitResult p1, FindSplitResult p2)
        {
            return p1.Equals(p2);
        }

        public static bool operator !=(FindSplitResult p1, FindSplitResult p2)
        {
            return !p1.Equals(p2);
        }

        public override int GetHashCode()
        {
            return BestSplitIndex.GetHashCode() ^
                BestInformationGain.GetHashCode() ^
                BestFeatureSplit.GetHashCode() ^
                LeftIntervalEntropy.GetHashCode() ^
                RightIntervalEntropy.GetHashCode();
        }

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
