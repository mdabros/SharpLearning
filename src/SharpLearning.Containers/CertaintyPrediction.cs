using System;

namespace SharpLearning.Containers;

/// <summary>
/// Certainty prediction for regression learners with certainty estimate
/// </summary>
[Serializable]
public struct CertaintyPrediction
{
    public readonly double Prediction;

    public readonly double Variance;

    /// <param name="prediction"></param>
    /// <param name="variance"></param>
    public CertaintyPrediction(double prediction, double variance)
    {
        Variance = variance;
        Prediction = prediction;
    }

    /// <param name="other"></param>
    /// <returns></returns>
    public bool Equals(CertaintyPrediction other)
    {
        if (!Equal(Prediction, other.Prediction)) { return false; }
        return Equal(Variance, other.Variance);
    }

    /// <param name="obj"></param>
    /// <returns></returns>
    public override bool Equals(object obj)
    {
        return obj is CertaintyPrediction prediction && Equals(prediction);
    }

    /// <param name="p1"></param>
    /// <param name="p2"></param>
    /// <returns></returns>
    public static bool operator ==(CertaintyPrediction p1, CertaintyPrediction p2)
    {
        return p1.Equals(p2);
    }

    /// <param name="p1"></param>
    /// <param name="p2"></param>
    /// <returns></returns>
    public static bool operator !=(CertaintyPrediction p1, CertaintyPrediction p2)
    {
        return !p1.Equals(p2);
    }

    /// <returns></returns>
    public override int GetHashCode()
    {
        return Prediction.GetHashCode() ^ Variance.GetHashCode();
    }

    const double Tolerence = 0.00001;

    static bool Equal(double a, double b)
    {
        var diff = Math.Abs(a * Tolerence);
        return Math.Abs(a - b) <= diff;
    }
}
