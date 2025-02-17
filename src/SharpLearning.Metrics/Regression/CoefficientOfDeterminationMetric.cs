﻿using System;
using System.Linq;

namespace SharpLearning.Metrics.Regression;

public sealed class CoefficientOfDeterminationMetric : IRegressionMetric
{
    /// <summary>
    /// Calculates coefficient of determination (r-squared) between the targets and predictions r-squared =  1 - sum((t-p)^2)/sum((t-t_mean)^2)
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

        var targetMean = targets.Sum() / targets.Length;
        var sStot = targets.Sum(target => Math.Pow(target - targetMean, 2));
        var sSres = 0.0;

        for (var i = 0; i < predictions.Length; i++)
        {
            sSres += Math.Pow(targets[i] - predictions[i], 2);
        }

        return sStot != 0.0 ? 1 - sSres / sStot : 0;
    }
}
