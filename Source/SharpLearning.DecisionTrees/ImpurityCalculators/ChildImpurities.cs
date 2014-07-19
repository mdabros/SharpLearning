using System;

namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    /// <summary>
    /// Struct for containing left and right child impurities
    /// </summary>
    public struct ChildImpurities : IEquatable<ChildImpurities>
    {
        public readonly double Left;
        public readonly double Right;

        public ChildImpurities(double left, double right)
        {
            Left = left;
            Right = right;
        }

        public bool Equals(ChildImpurities other)
        {
            if (!Equal(Left, other.Left)) { return false; }
            if (!Equal(Right, other.Right)) { return false; }

            return true;
        }

        public override bool Equals(object obj)
        {
            if (obj is ChildImpurities)
                return Equals((ChildImpurities)obj);
            return false;
        }


        public static bool operator ==(ChildImpurities p1, ChildImpurities p2)
        {
            return p1.Equals(p2);
        }

        public static bool operator !=(ChildImpurities p1, ChildImpurities p2)
        {
            return !p1.Equals(p2);
        }

        public override int GetHashCode()
        {
            return Left.GetHashCode() ^ Right.GetHashCode();
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
