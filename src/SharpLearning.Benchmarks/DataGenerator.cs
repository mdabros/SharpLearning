using System;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Benchmarks;

public static class DataGenerator
{
    // Default data size for benchmarks.
    const int Rows = 1000;
    const int Cols = 10;
    const int MinTargetValue = 0;
    const int MaxTargetValue = 10;
    const int Seed = 42;

    public static (F64Matrix Features, double[] Targets) GenerateRegressionData()
    {
        var random = new Random(Seed);
        var targets = GenerateDoubles(Rows, 1, random);
        var features = GenerateDoubles(Rows, Cols, random);
        return (new F64Matrix(features, Rows, Cols), targets);
    }

    public static (F64Matrix Features, double[] Targets) GenerateClassificationData()
    {
        var random = new Random(Seed);
        var targets = GenerateIntegers(Rows, 1, MinTargetValue, MaxTargetValue, random);
        var features = GenerateDoubles(Rows, Cols, random);
        return (new F64Matrix(features, Rows, Cols), targets);
    }

    static double[] GenerateDoubles(int rows, int cols, Random random)
    {
        var data = new double[rows * cols];
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble();
        }
        return data;
    }

    static double[] GenerateIntegers(int rows, int cols, int min, int max, Random random)
    {
        var data = new double[rows * cols];
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = random.Next(min, max);
        }
        return data;
    }
}
