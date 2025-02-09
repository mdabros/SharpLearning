﻿using System;

namespace SharpLearning.Metrics.Regression;

/// <summary>
/// Calculates the mean absolute error between the targets and predictions e = Sum(abs(t - p))/length(t) 
/// </summary>
public sealed class MeanAbsolutErrorRegressionMetric : IRegressionMetric
{
    /// <summary>
    /// Calculates the mean absolute error between the targets and predictions e = Sum(abs(t - p))/length(t) 
    /// </summary>
    /// <param name="targets"></param>
    /// <param name="predictions"></param>
    /// <returns></returns>
    public double Error(double[] targets, double[] predictions)
    {
        if (targets.Length != predictions.Length)
        {
            throw new ArgumentException("targets and predictions length do not match");
        }

        var meanSquareError = 0.0;
        for (int i = 0; i < targets.Length; ++i)
        {
            var targetValue = targets[i];
            var estimate = predictions[i];
            var error = Math.Abs(targetValue - estimate);
            meanSquareError += error;
        }
        meanSquareError *= (1.0 / targets.Length);

        return meanSquareError;
    }
}
