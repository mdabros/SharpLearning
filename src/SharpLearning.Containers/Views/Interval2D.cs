using System;
using System.Runtime.InteropServices;

namespace SharpLearning.Containers.Views;

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
        return (Cols.Equals(other.Cols)) &&
               (Rows.Equals(other.Rows));
    }

    public override bool Equals(object other)
    {
        return other is Interval2D interval2D && Equals(interval2D);
    }

    public static bool operator !=(Interval2D x, Interval2D y)
    {
        return !(x == y);
    }

    public static bool operator ==(Interval2D x, Interval2D y)
    {
        return (x.Cols == y.Cols) &&
               (x.Rows == y.Rows);
    }

    public override int GetHashCode()
    {
        return Cols.GetHashCode() ^ Rows.GetHashCode();
    }
}
