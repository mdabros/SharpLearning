
using System;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    /// <summary>
    /// Contains the the value and index created by a feature split
    /// </summary>
    public struct FeatureSplit : IEquatable<FeatureSplit>
    {
        public readonly double Threshold;
        public readonly int Index;

        public FeatureSplit(double value, int index)
        {
            this.Threshold = value;
            this.Index = index;
        }

        public bool Equals(FeatureSplit other)
        {
            if (Index != other.Index) { return false; }
            if (!Equal(Threshold, other.Threshold)) { return false; }

            return true;
        }

        public override bool Equals(object obj)
        {
            if (obj is FeatureSplit)
                return Equals((FeatureSplit)obj);
            return false;
        }


        public static bool operator ==(FeatureSplit p1, FeatureSplit p2)
        {
            return p1.Equals(p2);
        }

        public static bool operator !=(FeatureSplit p1, FeatureSplit p2)
        {
            return !p1.Equals(p2);
        }

        public override int GetHashCode()
        {
            return Index.GetHashCode() ^ Threshold.GetHashCode();
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
