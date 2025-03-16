using System;
using System.Linq;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Benchmarks;

public static class DataGenerator
{
    // default data size for benchmarks.
    const int Rows = 1000;
    const int Cols = 10;
    const int MinTargetValue = 0;
    const int MaxTargetValue = 10;
    const int Seed = 42;

    public static (F64Matrix Features, double[] Targets) GenerateRegressionData()
    {
        var targets = GenerateDoubles(Rows, cols: 1, Seed);
        var features = GenerateDoubles(Rows, Cols, Seed);
        return (new F64Matrix(features, Rows, Cols), targets);
    }

    public static (F64Matrix Features, double[] Targets) GenerateClassificationData()
    {
        var targets = GenerateIntegers(Rows, cols: 1, MinTargetValue, MaxTargetValue, Seed);
        var features = GenerateDoubles(Rows, Cols, Seed);
        return (new F64Matrix(features, Rows, Cols), targets);
    }

    static double[] GenerateDoubles(int rows, int cols, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, rows * cols)
            .Select(i => random.NextDouble()).ToArray();
    }

    static double[] GenerateIntegers(int rows, int cols, int min, int max, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, rows * cols)
            .Select(i => random.Next(min, max)).Select(i => (double)i).ToArray();
    }
}
