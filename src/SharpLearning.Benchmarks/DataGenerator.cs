using System;
using System.Linq;

namespace SharpLearning.Benchmarks;

public static class DataGenerator
{
    public static double[] GenerateDoubles(int rows, int cols, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, rows * cols)
            .Select(i => random.NextDouble()).ToArray();
    }

    public static double[] GenerateIntegers(int rows, int cols, int min, int max, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, rows * cols)
            .Select(i => random.Next(min, max)).Select(i => (double)i).ToArray();
    }
}
