﻿using System;
using System.Linq;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers;

/// <summary>
/// Container for storing an observations and targets pair.
/// </summary>
public sealed class ObservationTargetSet : IEquatable<ObservationTargetSet>
{
    public readonly F64Matrix Observations;

    public readonly double[] Targets;

    /// <summary>
    /// Container for storing an observations and targets pair.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    public ObservationTargetSet(F64Matrix observations, double[] targets)
    {
        Observations = observations ?? throw new ArgumentNullException(nameof(observations));
        Targets = targets ?? throw new ArgumentNullException(nameof(targets));
    }

    public bool Equals(ObservationTargetSet other)
    {
        if (!Observations.Equals(other.Observations)) { return false; }
        return Targets.SequenceEqual(other.Targets);
    }

    public override bool Equals(object obj)
    {
        return obj is ObservationTargetSet other && Equals(other);
    }

    public override int GetHashCode()
    {
        unchecked // Overflow is fine, just wrap
        {
            var hash = 17;
            hash = hash * 23 + Observations.GetHashCode();
            hash = hash * 23 + Targets.GetHashCode();

            return hash;
        }
    }
}
