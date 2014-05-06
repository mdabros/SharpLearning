using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using System.IO;

namespace SharpLearning.InputOutput.Csv
{
    public static class CsvRowExtensions
    {

        /// <summary>
        /// Gets the CsvRow value based on the supplied column name
        /// </summary>
        /// <param name="row"></param>
        /// <param name="columnNames"></param>
        /// <returns></returns>
        public static string GetValue(this CsvRow row, string columnName)
        {
            return row.Values[(row.ColumnNameToIndex[columnName])];
        }


        /// <summary>
        /// Gets the CsvRow values based on the supplied column names
        /// </summary>
        /// <param name="row"></param>
        /// <param name="columnNames"></param>
        /// <returns></returns>
        public static string[] GetValues(this CsvRow row, string[] columnNames)
        {
            var indices = columnNames.Select(n => row.ColumnNameToIndex[n]).ToArray();
            return row.Values.GetIndices(indices);
        }

        /// <summary>
        /// Parses the CsvRows to a double array. Only CsvRows with a single column can be used
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

            return dataRows.SelectMany(values => values.Values.AsF64()).ToArray();
        }

        /// <summary>
        /// Parses the CsvRows to a string array. Only CsvRows with a single column can be used
        /// </summary>
        /// <param name="dataRows"></param>
        /// <returns></returns>
        public static string[] ToStringVector(this IEnumerable<CsvRow> dataRows)
        {
            var first = dataRows.First();

            if (first.ColumnNameToIndex.Count != 1)
            {
                throw new ArgumentException("Vector can only be genereded from a single column");
            }

            return dataRows.SelectMany(values => values.Values).ToArray();
        }


        /// <summary>
        /// Parses the CsvRows to a F64Matrix
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
                return values.Values.AsF64();
            }).ToArray();

            return new F64Matrix(features, rows, cols);
        }

        /// <summary>
        /// Parses the CsvRows to a StringMatrix
        /// </summary>
        /// <param name="dataRows"></param>
        /// <returns></returns>
        public static StringMatrix ToStringMatrix(this IEnumerable<CsvRow> dataRows)
        {
            var first = dataRows.First();
            var cols = first.ColumnNameToIndex.Count;
            var rows = 0;

            var features = dataRows.SelectMany(values =>
            {
                rows++;
                return values.Values;
            }).ToArray();

            return new StringMatrix(features, rows, cols);
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

        public static void Write(this IEnumerable<CsvRow> dataRows, TextWriter writer, char separator = CsvParser.DefaultDelimiter)
        {
            using (var csvWriter = new CsvWriter(writer, separator))
            {
                csvWriter.Write(dataRows);
            }
        }
    }
}
