using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.InputOutput.Csv
{
    /// <summary>
    /// Extension methods for CsvRow
    /// </summary>
    public static class CsvRowExtensions
    {
        /// <summary>
        /// 
        /// </summary>
        public static readonly Converter<string,double> DefaultF64Converter = ArrayExtensions.DefaultF64Converter;

        /// <summary>
        /// Gets the CsvRow value based on the supplied column name
        /// </summary>
        /// <param name="row"></param>
        /// <param name="columnName"></param>
        /// <returns></returns>
        public static string GetValue(this CsvRow row, string columnName)
        {
            return row.Values[(row.ColumnNameToIndex[columnName])];
        }

        /// <summary>
        /// Sets the CsvRow value based on the supplied column name
        /// </summary>
        /// <param name="row"></param>
        /// <param name="columnName"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static void SetValue(this CsvRow row, string columnName, string value)
        {
            var index = row.ColumnNameToIndex[columnName];
            row.Values[index] = value;
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
        /// Keeps only the csv columns provided in columnNames
        /// </summary>
        /// <param name="dataRows"></param>
        /// <param name="columnNames"></param>
        /// <returns></returns>
        public static IEnumerable<CsvRow> Keep(this IEnumerable<CsvRow> dataRows, params string[] columnNames)
        {
            var index = 0;
            var reducedColumnNameToIndex = columnNames.ToDictionary(n => n, n => index++);

            foreach (var row in dataRows)
            {
                yield return new CsvRow(reducedColumnNameToIndex, row.GetValues(columnNames));
            }
        }

        /// <summary>
        /// Removes the csv columns provided in columnNames
        /// </summary>
        /// <param name="dataRows"></param>
        /// <param name="columnNames"></param>
        /// <returns></returns>
        public static IEnumerable<CsvRow> Remove(this IEnumerable<CsvRow> dataRows, params string[] columnNames)
        {
            var columnsToKeep = dataRows.First().ColumnNameToIndex.Keys.Except(columnNames).ToArray();
            var index = 0;
            var reducedColumnNameToIndex = columnsToKeep.ToDictionary(n => n, n => index++);

            foreach (var row in dataRows)
            {
                yield return new CsvRow(reducedColumnNameToIndex, row.GetValues(columnsToKeep));
            }
        }

        /// <summary>
        /// Parses the CsvRows to a double array. Only CsvRows with a single column can be used
        /// </summary>
        /// <param name="dataRows"></param>
        /// <returns></returns>
        public static double[] ToF64Vector(this IEnumerable<CsvRow> dataRows)
        {
            return ToF64Vector(dataRows, DefaultF64Converter);
        }

        /// <summary>
        /// Parses the CsvRows to a double array. Only CsvRows with a single column can be used
        /// </summary>
        /// <param name="dataRows"></param>
        /// <param name="converter"></param>
        /// <returns></returns>
        public static double[] ToF64Vector(this IEnumerable<CsvRow> dataRows,
            Converter<string, double> converter)
        {
            var first = dataRows.First();

            if (first.ColumnNameToIndex.Count != 1)
            {
                throw new ArgumentException("Vector can only be genereded from a single column");
            }

            return dataRows.SelectMany(values => values.Values.AsF64(converter)).ToArray();
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
                throw new ArgumentException("Vector can only be generated from a single column");
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
            return ToF64Matrix(dataRows, DefaultF64Converter);
        }

        /// <summary>
        /// Parses the CsvRows to a F64Matrix
        /// </summary>
        /// <param name="dataRows"></param>
        /// <param name="converter"></param>
        /// <returns></returns>
        public static F64Matrix ToF64Matrix(this IEnumerable<CsvRow> dataRows,
            Converter<string, double> converter)
        {
            var first = dataRows.First();
            var cols = first.ColumnNameToIndex.Count;
            var rows = 0;

            var features = dataRows.SelectMany(values =>
            {
                rows++;
                return values.Values.AsF64(converter);
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
        public static IEnumerable<CsvRow> EnumerateCsvRows<T>(this IMatrix<T> matrix, 
            Dictionary<string, int> columnNameToIndex)
        {
            var rows = matrix.RowCount;
            var cols = matrix.ColumnCount;

            if (cols != columnNameToIndex.Count)
            {
                throw new ArgumentException("matrix and column name to index must have same lengths");
            }

            for (int i = 0; i < rows; i++)
            {
                var row = matrix.Row(i)
                                .Select(value => value.ToString())
                                .ToArray();

                yield return new CsvRow(columnNameToIndex, row);
            }
        }

        /// <summary>
        /// Writes the CsvRows to the provided stream
        /// </summary>
        /// <param name="dataRows"></param>
        /// <param name="writer"></param>
        /// <param name="separator"></param>
        /// <param name="writeHeader">True and a header is added to the stream, false and the header is omitted</param>
        public static void Write(this IEnumerable<CsvRow> dataRows, 
            Func<TextWriter> writer, 
            char separator = CsvParser.DefaultDelimiter, 
            bool writeHeader = true)
        {
            new CsvWriter(writer, separator).Write(dataRows, writeHeader);
        }

        /// <summary>
        /// Writes the CsvRows to file path
        /// </summary>
        /// <param name="dataRows"></param>
        /// <param name="filePath"></param>
        /// <param name="separator"></param>
        /// <param name="writeHeader">True and a header is added to the stream, false and the header is omitted</param>
        public static void WriteFile(this IEnumerable<CsvRow> dataRows, 
            string filePath, 
            char separator = CsvParser.DefaultDelimiter, 
            bool writeHeader = true)
        {
            Write(dataRows, () => new StreamWriter(filePath), separator, writeHeader);
        }

        /// <summary>
        /// Combines two IEnumerables based on column header names. Matching rows are combined and parsed on. 
        /// </summary>
        /// <param name="thisRows"></param>
        /// <param name="otherRows"></param>
        /// <param name="key1"></param>
        /// <param name="key2"></param>
        /// <param name="removeRepeatedColumns">Should repeated columns be removed</param>
        /// <returns></returns>
        public static IEnumerable<CsvRow> KeyCombine(this IEnumerable<CsvRow> thisRows,
            IEnumerable<CsvRow> otherRows,
            string key1,
            string key2,
            bool removeRepeatedColumns = true)
        {
            CreateNewColumnNameToIndex(thisRows, otherRows, removeRepeatedColumns,
                out Dictionary<string, int> newColumnNameToIndex, out int[] columnIndicesToKeep);

            var dictThisRows = thisRows.ToDictionary(p => p.GetValue(key1));
            var dictOtherRows = otherRows.ToDictionary(p => p.GetValue(key2));

            foreach (var key in dictThisRows.Keys)
            {
                if (!dictOtherRows.ContainsKey(key)) continue;

                var thisValues = dictThisRows[key].Values;
                var otherValues = dictOtherRows[key].Values;

                if (!removeRepeatedColumns)
                {
                    var newValues = new string[thisValues.Length + otherValues.Length];

                    thisValues.CopyTo(newValues, 0);
                    otherValues.CopyTo(newValues, thisValues.Length);

                    yield return new CsvRow(newColumnNameToIndex, newValues);
                }
                else
                {
                    var newValues = new string[newColumnNameToIndex.Count];
                    var reducedOtherValues = otherValues.GetIndices(columnIndicesToKeep);

                    thisValues.CopyTo(newValues, 0);
                    reducedOtherValues.CopyTo(newValues, thisValues.Length);

                    yield return new CsvRow(newColumnNameToIndex, newValues);
                }
            }
        }

        /// <summary>
        /// Combines two IEnumerables based on a row matcher function. Matching rows are combined and parsed on. 
        /// </summary>
        /// <param name="thisRows"></param>
        /// <param name="otherRows"></param>
        /// <param name="rowMatcher"></param>
        /// <param name="removeRepeatedColumns">Should repeated columns be removed</param>
        /// <returns></returns>
        public static IEnumerable<CsvRow> KeyCombine(this IEnumerable<CsvRow> thisRows,
            IEnumerable<CsvRow> otherRows,
            Func<CsvRow, CsvRow, bool> rowMatcher,
            bool removeRepeatedColumns = true)
        {
            CreateNewColumnNameToIndex(thisRows, otherRows, removeRepeatedColumns,
                out Dictionary<string, int> newColumnNameToIndex, out int[] columnIndicesToKeep);

            foreach (var thisRow in thisRows)
            {
                foreach (var otherRow in otherRows)
                {
                    var thisValues = thisRow.Values;
                    var otherValues = otherRow.Values;

                    if (rowMatcher(thisRow, otherRow))
                    {
                        if (!removeRepeatedColumns)
                        {
                            var newValues = new string[thisValues.Length + otherValues.Length];

                            thisValues.CopyTo(newValues, 0);
                            otherValues.CopyTo(newValues, thisValues.Length);

                            yield return new CsvRow(newColumnNameToIndex, newValues);
                            break;
                        }
                        else
                        {
                            var newValues = new string[newColumnNameToIndex.Count];
                            var reducedOtherValues = otherValues.GetIndices(columnIndicesToKeep);

                            thisValues.CopyTo(newValues, 0);
                            reducedOtherValues.CopyTo(newValues, thisValues.Length);

                            yield return new CsvRow(newColumnNameToIndex, newValues);
                            break;
                        }
                    }
                }
            }
        }

        static void CreateNewColumnNameToIndex(IEnumerable<CsvRow> thisRows,
            IEnumerable<CsvRow> otherRows,
            bool removeRepeatedColumns,
            out Dictionary<string, int> newColumnNameToIndex,
            out int[] columnIndicesToKeep)
        {
            newColumnNameToIndex = thisRows.First().ColumnNameToIndex.ToDictionary(k => k.Key, k => k.Value);
            var otherColumnNameToIndex = otherRows.First().ColumnNameToIndex;
            var columnIndicesToRemove = new List<int>();

            foreach (var kvp in otherColumnNameToIndex)
            {
                if (newColumnNameToIndex.ContainsKey(kvp.Key))
                {
                    if (!removeRepeatedColumns)
                    {
                        newColumnNameToIndex.Add(CreateKey(kvp.Key, newColumnNameToIndex), newColumnNameToIndex.Count);
                    }
                    else
                    {
                        columnIndicesToRemove.Add(kvp.Value);
                    }
                }
                else
                {
                    newColumnNameToIndex.Add(kvp.Key, newColumnNameToIndex.Count);
                }
            }

            columnIndicesToKeep = otherColumnNameToIndex.Values.Except(columnIndicesToRemove).ToArray();
        }

        static string CreateKey(string key, Dictionary<string, int> columnNameToIndex)
        {
            if (!columnNameToIndex.ContainsKey(key))
            {
                return key;
            }
            else
            {
                var index = 1;
                var newKey = key + "_" + index;
                while (columnNameToIndex.ContainsKey(newKey))
                {
                    index++;
                    newKey = key + "_" + index;
                }
                return newKey;
            }
        }
    }
}
