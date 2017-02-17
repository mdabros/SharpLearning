using System;
using System.Runtime.InteropServices;

namespace SharpLearning.Containers.Views
{
    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Interval2D : IEquatable<Interval2D>
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly Interval1D Rows;

        /// <summary>
        /// 
        /// </summary>
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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(Interval2D other)
        {
            return (this.Cols.Equals(other.Cols)) &&
                   (this.Rows.Equals(other.Rows));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public override bool Equals(object other)
        {
            if (other is Interval2D)
                return this.Equals((Interval2D)other);
            else
                return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static bool operator !=(Interval2D x, Interval2D y)
        { 
            return !(x == y); 
        }
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static bool operator ==(Interval2D x, Interval2D y)
        { 
            return (x.Cols == y.Cols) && 
                   (x.Rows== y.Rows); 
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return this.Cols.GetHashCode() ^ this.Rows.GetHashCode();
        }
    }
}
