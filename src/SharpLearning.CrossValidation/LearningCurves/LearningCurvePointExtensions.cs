using System;
using System.Collections.Generic;
using System.IO;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.CrossValidation.LearningCurves;

/// <summary>
/// Extension methods for LearningCurvePoint
/// </summary>
public static class LearningCurvePointExtensions
{
    /// <summary>
    /// Converts a list of n LearningCurvePoint to a n by 3 matrix
    /// with columns SampleSize, TrainingScore and ValidationScore.
    /// </summary>
    /// <param name="points"></param>
    /// <returns></returns>
    public static F64Matrix ToF64Matrix(this List<LearningCurvePoint> points)
    {
        if (points.Count == 0)
        { throw new ArgumentException("There must be at least one element in the list to convert to a matrix"); }

        var matrix = new F64Matrix(points.Count, 3);
        for (var i = 0; i < points.Count; i++)
        {
            var point = points[i];
            matrix.At(i, 0, point.SampleSize);
            matrix.At(i, 1, point.TrainingScore);
            matrix.At(i, 2, point.ValidationScore);
        }

        return matrix;
    }

    /// <summary>
    /// Writes the list of BiasVarianceLearningCurvePoint as csv to the provided writer
    /// </summary>
    /// <param name="points"></param>
    /// <param name="writer"></param>
    /// <param name="separator"></param>
    public static void Write(this List<LearningCurvePoint> points, Func<TextWriter> writer,
        char separator = CsvParser.DefaultDelimiter)
    {
        var columnNameToIndex = new Dictionary<string, int>
        {
            { "SampleCount", 0 },
            { "TrainingError", 1 },
            { "ValidationError", 2 }
        };

        points.ToF64Matrix()
            .EnumerateCsvRows(columnNameToIndex)
            .Write(writer, separator);
    }

    /// <summary>
    /// Writes the list of BiasVarianceLearningCurvePoint as csv to file path
    /// </summary>
    /// <param name="points"></param>
    /// <param name="filePath"></param>
    /// <param name="separator"></param>
    public static void WriteFile(this List<LearningCurvePoint> points, string filePath,
        char separator = CsvParser.DefaultDelimiter)
    {
        Write(points, () => new StreamWriter(filePath), separator);
    }
}
