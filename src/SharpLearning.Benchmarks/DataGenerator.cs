using System;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Benchmarks;

public static class DataGenerator
{
    // Default data size for benchmarks.
    public const int DefaultRows = 1000;
    public const int DefaultCols = 10;
    public const int DefaultMinTargetValue = 0;
    public const int DefaultMaxTargetValue = 10;
    public const int DefaultSeed = 42;

    public static (F64Matrix Features, double[] Targets) GenerateRegressionData(
        int rows = DefaultRows, int cols = DefaultCols,
        int seed = DefaultSeed)
    {
        var random = new Random(seed);
        var features = GenerateRandomDoubles(rows, cols, random);
        var targets = GenerateRandomDoubles(rows, 1, random);
        return (new F64Matrix(features, rows, cols), targets);
    }

    public static (F64Matrix Features, double[] Targets) GenerateClassificationData(
        int rows = DefaultRows, int cols = DefaultCols,
        int minTargetValue = DefaultMinTargetValue,
        int maxTargetValue = DefaultMaxTargetValue,
        int seed = DefaultSeed)
    {
        var random = new Random(seed);
        var features = GenerateRandomDoubles(rows, cols, random);
        var targets = GenerateRandomIntegers(rows, 1, minTargetValue, maxTargetValue, random);
        return (new F64Matrix(features, rows, cols), targets);
    }

    static double[] GenerateRandomDoubles(int rows, int cols, Random random)
    {
        var data = new double[rows * cols];
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble();
        }
        return data;
    }

    static double[] GenerateRandomIntegers(int rows, int cols, int min, int max, Random random)
    {
        var data = new double[rows * cols];
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = random.Next(min, max);
        }
        return data;
    }
}
