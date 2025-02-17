﻿using System;
using System.Runtime.InteropServices;

namespace SharpLearning.Containers.Views;

[StructLayout(LayoutKind.Sequential)]
public struct Interval1D : IEquatable<Interval1D>
{
    public readonly int FromInclusive;

    public readonly int ToExclusive;

    public readonly int Length;

    /// <summary>
    /// Creates a 1D interval as specified from inclusive to exclusive
    /// </summary>
    /// <param name="fromInclusive"></param>
    /// <param name="toExclusive"></param>
    public Interval1D(int fromInclusive, int toExclusive)
    {
        if (fromInclusive >= toExclusive)
        {
            throw new ArgumentException($"FromInclusive: {fromInclusive}" +
                "is larger or equal to toExclusive: {toExclusive}");
        }
        FromInclusive = fromInclusive;
        ToExclusive = toExclusive;
        Length = toExclusive - fromInclusive;
    }

    /// <summary>
    /// Creates a 1D interval as specified from inclusive to exclusive
    /// </summary>
    /// <param name="fromInclusive"></param>
    /// <param name="toExclusive"></param>
    public static Interval1D Create(int fromInclusive, int toExclusive)
    {
        return new Interval1D(fromInclusive, toExclusive);
    }

    public static bool operator !=(Interval1D x, Interval1D y)
    {
        return !(x == y);
    }

    public static bool operator ==(Interval1D x, Interval1D y)
    {
        return (x.FromInclusive == y.FromInclusive) &&
               (x.ToExclusive == y.ToExclusive);
    }

    public bool Equals(Interval1D other)
    {
        return (FromInclusive == other.FromInclusive) &&
               (ToExclusive == other.ToExclusive) &&
               (Length == other.Length);
    }

    public override bool Equals(object other)
    {
        return other is Interval1D interval1D && Equals(interval1D);
    }

    public override int GetHashCode()
    {
        return FromInclusive.GetHashCode() ^ ToExclusive.GetHashCode() ^ Length.GetHashCode();
    }
}
