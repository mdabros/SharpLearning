using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.InputOutput.Csv
{
    public static class CsvRowExtensions
    {

        /// <summary>
        /// Combines two IEnumerables based on a row matcher function. Matching rows are combined and parsed on. 
        /// </summary>
        /// <param name="thisRows"></param>
        /// <param name="otherRows"></param>
        /// <param name="rowMatcher"></param>
        /// <returns></returns>
        public static IEnumerable<CsvRow> KeyCombine(this IEnumerable<CsvRow> thisRows, IEnumerable<CsvRow> otherRows, Func<CsvRow, CsvRow, bool> rowMatcher, bool removeRepeatedColumns=true)
        {
            var newColumnNameToIndex = thisRows.First().ColumnNameToIndex.ToDictionary(k => k.Key, k => k.Value);
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

            var columnIndicesToKeep = otherColumnNameToIndex.Values
                .Except(columnIndicesToRemove).ToArray();

            foreach (var thisRow in thisRows)
            {
                foreach (var otherRow in otherRows)
                {
                    var thisValues = thisRow.Values;
                    var otherValues = otherRow.Values;

                    if(rowMatcher(thisRow, otherRow))
                    {
                        if(!removeRepeatedColumns)
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

        static string CreateKey(string key, Dictionary<string, int> columnNameToIndex)
        {
            if(!columnNameToIndex.ContainsKey(key))
            {
                return key;
            }
            else
            {
                var index = 1;
                var newKey = key + "_" + index;
                while(columnNameToIndex.ContainsKey(newKey))
                {
                    index++;
                    newKey = key + "_" + index;
                }
                return newKey;
            }
        }

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

                yield return new CsvRow(columnNameToIndex, row);
            }
        }

        /// <summary>
        /// Writes the CsvRows to the provided stream
        /// </summary>
        /// <param name="dataRows"></param>
        /// <param name="writer"></param>
        /// <param name="separator"></param>
        /// <param name="writeHeader">True and a header is added to the stream, false and the header is omittet</param>
        public static void Write(this IEnumerable<CsvRow> dataRows, Func<TextWriter> writer, char separator = CsvParser.DefaultDelimiter, bool writeHeader = true)
        {
            new CsvWriter(writer, separator).Write(dataRows, writeHeader);
        }

        /// <summary>
        /// Writes the CsvRows to file path
        /// </summary>
        /// <param name="dataRows"></param>
        /// <param name="filePath"></param>
        /// <param name="separator"></param>
        /// <param name="writeHeader">True and a header is added to the stream, false and the header is omittet</param>
        public static void WriteFile(this IEnumerable<CsvRow> dataRows, string filePath, char separator = CsvParser.DefaultDelimiter, bool writeHeader = true)
        {
            Write(dataRows, () => new StreamWriter(filePath), separator, writeHeader);
        }
    }
}
