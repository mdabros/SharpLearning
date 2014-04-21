using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.InputOutput.Csv
{
    public static class CsvRowExtensions
    {
        /// <summary>
        /// Parses the CsvRows to a vector. Only CsvRows with a single column can be used
        /// </summary>
        /// <param name="dataRows"></param>
        /// <returns></returns>
        public static double[] ToF64Vector(this IEnumerable<CsvRow> dataRows)
        {
            var first = dataRows.First();

            if (first.ColumnNameToIndex.Count != 1)
            {
                throw new ArgumentException("Vector can only be genereded from a single column");
            }

            return dataRows.SelectMany(values => values.Values.Select(v => FloatingPointConversion.ToF64(v))).ToArray();
        }

        /// <summary>
        /// Parses the CsvRows to an F64Matrix
        /// </summary>
        /// <param name="dataRows"></param>
        /// <returns></returns>
        public static F64Matrix ToF64Matrix(this IEnumerable<CsvRow> dataRows)
        {
            var first = dataRows.First();
            var cols = first.ColumnNameToIndex.Count;
            var rows = 0;

            var features = dataRows.SelectMany(values =>
            {
                rows++;
                return values.Values.Select(v => FloatingPointConversion.ToF64(v));
            }).ToArray();

            return new F64Matrix(features, rows, cols);
        }

        /// <summary>
        /// Enumerates a Matrix to CsvRows. 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="matrix"></param>
        /// <param name="columnNameToIndex"></param>
        /// <returns></returns>
        public static IEnumerable<CsvRow> EnumerateCsvRows<T>(this IMatrix<T> matrix, Dictionary<string, int> columnNameToIndex)
        {
            var rows = matrix.GetNumberOfRows();
            var cols = matrix.GetNumberOfColumns();

            if (cols != columnNameToIndex.Count)
            {
                throw new ArgumentException("matrix and column name to index must have same lengths");
            }

            for (int i = 0; i < rows; i++)
            {
                var row = matrix.GetRow(i)
                                .Select(value => value.ToString())
                                .ToArray();

                yield return new CsvRow(row, columnNameToIndex);
            }
        }

        //public static void Write(this IEnumerable<CsvRow> dataRows, TextWriter writer, char separator = CsvParser.DefaultDelimiter)
        //{
        //    using (var csvWriter = new CsvWriter(writer, separator))
        //    {
        //        csvWriter.Write(dataRows);
        //    }
        //}
    }
}
