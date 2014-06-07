using System;
using System.Runtime.InteropServices;

namespace SharpLearning.Containers.Views
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Interval2D : IEquatable<Interval2D>
    {
        public readonly Interval1D Rows;
        public readonly Interval1D Cols;

        /// <summary>
        /// Creates a 2D interval based on the provided row and column intervals
        /// </summary>
        /// <param name="rowInterval"></param>
        /// <param name="colInterval"></param>
        public Interval2D(Interval1D rowInterval, Interval1D colInterval)
        {
            Rows = rowInterval;
            Cols = colInterval;
        }

        /// <summary>
        /// Creates a 2D interval based on the provided row and column intervals
        /// </summary>
        /// <param name="rowInterval"></param>
        /// <param name="colInterval"></param>
        public static Interval2D Create(Interval1D rowInterval, Interval1D colInterval)
        {
            return new Interval2D(rowInterval, colInterval);
        }

        public bool Equals(Interval2D other)
        {
            return (this.Cols.Equals(other.Cols)) &&
                   (this.Rows.Equals(other.Rows));
        }

        public override bool Equals(object other)
        {
            if (other is Interval2D)
                return this.Equals((Interval2D)other);
            else
                return false;
        }

        public static bool operator !=(Interval2D x, Interval2D y)
        { 
            return !(x == y); 
        }
        
        public static bool operator ==(Interval2D x, Interval2D y)
        { 
            return (x.Cols == y.Cols) && 
                   (x.Rows== y.Rows); 
        }

        public override int GetHashCode()
        {
            return this.Cols.GetHashCode() ^ this.Rows.GetHashCode();
        }
    }
}
