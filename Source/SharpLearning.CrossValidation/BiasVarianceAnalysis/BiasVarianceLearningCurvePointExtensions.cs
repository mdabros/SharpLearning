using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.CrossValidation.BiasVarianceAnalysis
{
    /// <summary>
    /// Extension methods for BiasVarianceLearningCurvePoint
    /// </summary>
    public static class BiasVarianceLearningCurvePointExtensions
    {
        /// <summary>
        /// Converts a list of n BiasVarianceLearningCurvePoint to a n by 3 matrix
        /// with columns SampleSize, TrainingScore and ValidationScore.
        /// </summary>
        /// <param name="points"></param>
        /// <returns></returns>
        public static F64Matrix ToF64Matrix(this List<BiasVarianceLearningCurvePoint> points)
        {
            if (points.Count == 0)
            { throw new ArgumentException("There must be at least one element in the list to convert to a matrix"); };

            var matrix = new F64Matrix(points.Count, 3);
            for (int i = 0; i < points.Count; i++)
            {
                var point = points[i];
                matrix.SetItemAt(i, 0, point.SampleSize);
                matrix.SetItemAt(i, 1, point.TrainingScore);
                matrix.SetItemAt(i, 2, point.ValidationScore);
            }

            return matrix;
        }

        public static void Write(this List<BiasVarianceLearningCurvePoint> points, Func<TextWriter> writer, 
            char separator = CsvParser.DefaultDelimiter)
        {
            points.ToF64Matrix()
                .EnumerateCsvRows(new Dictionary<string, int> { { "SampleCount", 0 }, { "TrainingError", 1 }, { "TestError", 2 } })
                .Write(writer, separator);
        }
    }
}
