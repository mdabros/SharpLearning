using System;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.CrossValidation.TrainingTestSplitters;

/// <summary>
/// Container for storing training set/test set split
/// </summary>
public sealed class TrainingTestSetSplit : IEquatable<TrainingTestSetSplit>
{
    /// <summary>
    ///
    /// </summary>
    public readonly ObservationTargetSet TrainingSet;

    /// <summary>
    ///
    /// </summary>
    public readonly ObservationTargetSet TestSet;

    /// <summary>
    /// Container for storing training set/test set split.
    /// </summary>
    /// <param name="trainingSet"></param>
    /// <param name="testSet"></param>
    public TrainingTestSetSplit(ObservationTargetSet trainingSet, ObservationTargetSet testSet)
    {
        TrainingSet = trainingSet ?? throw new ArgumentNullException(nameof(trainingSet));
        TestSet = testSet ?? throw new ArgumentNullException(nameof(testSet));
    }

    /// <summary>
    /// Container for storing training set/test set split.
    /// </summary>
    /// <param name="trainingObservations"></param>
    /// <param name="trainingTargets"></param>
    /// <param name="testObservations"></param>
    /// <param name="testTargets"></param>
    public TrainingTestSetSplit(F64Matrix trainingObservations, double[] trainingTargets,
        F64Matrix testObservations, double[] testTargets)
        : this(new ObservationTargetSet(trainingObservations, trainingTargets),
        new ObservationTargetSet(testObservations, testTargets))
    {
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public bool Equals(TrainingTestSetSplit other)
    {
        if (!TrainingSet.Equals(other.TrainingSet)) { return false; }
        return TestSet.Equals(other.TestSet);
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="obj"></param>
    /// <returns></returns>
    public override bool Equals(object obj)
    {
        return obj is TrainingTestSetSplit other && Equals(other);
    }

    /// <summary>
    ///
    /// </summary>
    /// <returns></returns>
    public override int GetHashCode()
    {
        unchecked // Overflow is fine, just wrap
        {
            var hash = 17;
            hash = hash * 23 + TrainingSet.GetHashCode();
            hash = hash * 23 + TestSet.GetHashCode();

            return hash;
        }
    }
}
