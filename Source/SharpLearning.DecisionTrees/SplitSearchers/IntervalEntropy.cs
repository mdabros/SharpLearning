using SharpLearning.Containers.Views;
using System;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    /// <summary>
    /// Structure for holding a grouping an interval with its entropy
    /// </summary>
    public struct IntervalEntropy : IEquatable<IntervalEntropy>
    {
        public readonly Interval1D Interval;
        public readonly double Entropy;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="interval"></param>
        /// <param name="entropy"></param>
        public IntervalEntropy(Interval1D interval, double entropy)
        {
            Interval = interval;
            Entropy = entropy;
        }

        /// <summary>
        /// Returns an initial IntervalEntropy with start values 
        /// </summary>
        /// <returns></returns>
        public static IntervalEntropy Initial()
        {
            return new IntervalEntropy(Interval1D.Create(0, 1), 0.0);
        }

        public bool Equals(IntervalEntropy other)
        {
            if (!Interval.Equals(other.Interval)) { return false; }
            if (!Equal(Entropy, other.Entropy)) { return false; }

            return true;
        }

        public override bool Equals(object obj)
        {
            if (obj is IntervalEntropy)
                return Equals((IntervalEntropy)obj);
            return false;
        }


        public static bool operator ==(IntervalEntropy p1, IntervalEntropy p2)
        {
            return p1.Equals(p2);
        }

        public static bool operator !=(IntervalEntropy p1, IntervalEntropy p2)
        {
            return !p1.Equals(p2);
        }

        public override int GetHashCode()
        {
            return Interval.GetHashCode() ^ Entropy.GetHashCode();
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
