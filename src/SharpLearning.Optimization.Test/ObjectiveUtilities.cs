using System;

namespace SharpLearning.Optimization.Test;

public static class ObjectiveUtilities
{
    public const double Delta = 0.000001;

    public static OptimizerResult Minimize(double[] x)
    {
        return new OptimizerResult(x, Math.Sin(x[0]) * Math.Cos(x[1]) * (1.0 / (Math.Abs(x[2]) + 1)));
    }

    public static OptimizerResult MinimizeWeightFromHeight(double[] parameters)
    {
        var heights = new double[] { 1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83 };
        var weights = new double[] { 52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46 };

        var cost = 0.0;

        for (var i = 0; i < heights.Length; i++)
        {
            cost += (parameters[0] * heights[i] - weights[i]) * (parameters[0] * heights[i] - weights[i]);
        }

        return new OptimizerResult(parameters, cost);
    }

    public static OptimizerResult MinimizeNonDeterministic(double[] x, Random random)
    {
        //less than 1 has lower reward
        return new OptimizerResult(x, x[0] < 1 ? random.NextDouble() : 1);
    }
}
