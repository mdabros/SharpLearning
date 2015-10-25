using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;
using System;
using System.Collections.Generic;

namespace SharpLearning.FeatureTransformations
{
    /// <summary>
    /// Contains extension methods for applying feature transforms to IEnumerable<CsvRow>.
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
    }
}
