using System;

namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    /// <summary>
    /// Struct for containing left and right child impurities
    /// </summary>
    public struct ChildImpurities : IEquatable<ChildImpurities>
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly double Left;

        /// <summary>
        /// 
        /// </summary>
        public readonly double Right;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        public ChildImpurities(double left, double right)
        {
            Left = left;
            Right = right;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(ChildImpurities other)
        {
            if (!Equal(Left, other.Left)) { return false; }
            if (!Equal(Right, other.Right)) { return false; }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (obj is ChildImpurities)
                return Equals((ChildImpurities)obj);
            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator ==(ChildImpurities p1, ChildImpurities p2)
        {
            return p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator !=(ChildImpurities p1, ChildImpurities p2)
        {
            return !p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
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
