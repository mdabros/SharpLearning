using System;
using System.Collections.Generic;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations
{
    /// <summary>
    /// Contains extension methods for applying feature transforms to CsvRows.
    /// </summary>
    public static class FeatureTransformationExtensions
    {
        /// <summary>
        /// Makes it possible to fluently apply a series of feature transformations
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="transformFunc"></param>
        /// <returns></returns>
        public static IEnumerable<CsvRow> Transform(this IEnumerable<CsvRow> rows, 
            Func<IEnumerable<CsvRow>, IEnumerable<CsvRow>> transformFunc)
        {
            return transformFunc(rows);
        }

        /// <summary>
        /// Transforms the matrix using the transform function. Values in matrix are replaced.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="transformFunc"></param>
        /// <returns></returns>
        public static StringMatrix Transform(this StringMatrix matrix, 
            Action<StringMatrix, StringMatrix> transformFunc)
        {
            transformFunc(matrix, matrix);
            return matrix;
        }

        /// <summary>
        /// Transforms the matrix using the transform function. Values in matrix are replaced.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="transformFunc"></param>
        /// <returns></returns>
        public static F64Matrix Transform(this F64Matrix matrix, 
            Action<F64Matrix, F64Matrix> transformFunc)
        {
            transformFunc(matrix, matrix);
            return matrix;
        }
    }
}
