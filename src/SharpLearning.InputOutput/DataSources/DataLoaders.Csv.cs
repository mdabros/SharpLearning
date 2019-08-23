using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.InputOutput.DataSources
{
    /// <summary>
    /// Factory methods for creating data loaders.
    /// </summary>
    public static partial class DataLoaders
    {
        /// <summary>
        /// Load data from Csv sources.
        /// </summary>
        public static class Csv
        {
            /// <summary>
            /// Creates a DataLoader reading from Csv Data.
            /// </summary>
            /// <typeparam name="T"></typeparam>
            /// <param name="csvText"></param>
            /// <param name="columnParser"></param>
            /// <param name="selectColumnNames"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<T> FromText<T>(string csvText,
                Func<string, T> columnParser,
                Func<string, bool> selectColumnNames,
                int[] sampleShape)
            {
                var parser = CsvParser.FromText(csvText);
                return FromParser(parser, columnParser, selectColumnNames, sampleShape);
            }

            /// <summary>
            /// Creates a DataLoader reading from Csv Data.
            /// </summary>
            /// <typeparam name="T"></typeparam>
            /// <param name="csvText"></param>
            /// <param name="columnParser"></param>
            /// <param name="columnNames"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<T> FromText<T>(string csvText,
                Func<string, T> columnParser,
                string[] columnNames,
                int[] sampleShape)
            {
                var parser = CsvParser.FromText(csvText);
                return FromParser(parser, columnParser, columnNames, sampleShape);
            }

            /// <summary>
            /// Creates a DataLoader reading from Csv Data.
            /// </summary>
            /// <typeparam name="T"></typeparam>
            /// <param name="csvFilePath"></param>
            /// <param name="columnParser"></param>
            /// <param name="selectColumnNames"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<T> FromFile<T>(string csvFilePath,
                Func<string, T> columnParser,
                Func<string, bool> selectColumnNames,
                int[] sampleShape)
            {
                var parser = CsvParser.FromFile(csvFilePath);
                return FromParser(parser, columnParser, selectColumnNames, sampleShape);
            }

            /// <summary>
            /// Creates a DataLoader reading from Csv Data.
            /// </summary>
            /// <typeparam name="T"></typeparam>
            /// <param name="csvFilePath"></param>
            /// <param name="columnParser"></param>
            /// <param name="columnNames"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<T> FromFile<T>(string csvFilePath,
                Func<string, T> columnParser,
                string[] columnNames,
                int[] sampleShape)
            {
                var parser = CsvParser.FromFile(csvFilePath);
                return FromParser(parser, columnParser, columnNames, sampleShape);
            }

            /// <summary>
            /// Creates a DataLoader reading from Csv Data.
            /// </summary>
            /// <typeparam name="T"></typeparam>
            /// <param name="parser"></param>
            /// <param name="columnParser"></param>
            /// <param name="selectColumnNames"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<T> FromParser<T>(CsvParser parser,
                Func<string, T> columnParser,
                Func<string, bool> selectColumnNames,
                int[] sampleShape)
            {
                var rows = parser.EnumerateRows(selectColumnNames);
                var totalSampleCount = rows.Count();
                return FromCsvRows<T>(rows, columnParser, sampleShape, totalSampleCount);
            }

            /// <summary>
            /// Creates a DataLoader reading from Csv Data.
            /// </summary>
            /// <typeparam name="T"></typeparam>
            /// <param name="parser"></param>
            /// <param name="columnParser"></param>
            /// <param name="columnNames"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<T> FromParser<T>(CsvParser parser,
                Func<string, T> columnParser,
                string[] columnNames,
                int[] sampleShape)
            {
                var rows = parser.EnumerateRows(columnNames);
                var totalSampleCount = rows.Count();
                return FromCsvRows<T>(rows, columnParser, sampleShape, totalSampleCount);
            }

            static DataLoader<T> FromCsvRows<T>(
                IEnumerable<CsvRow> rows,
                Func<string, T> columnParser,
                int[] sampleShape,
                int totalSampleCount)
            {
                var sampleSize = sampleShape.Aggregate((v1, v2) => v1 * v2);

                DataBatch<T> LoadCsvData(int[] indices)
                {
                    var batchSampleCount = indices.Length;
                    var data = new T[batchSampleCount * sampleSize];
                    var currentIndex = 0;
                    var copyIndexStart = 0;

                    foreach (var row in rows)
                    {
                        if (indices.Contains(currentIndex))
                        {
                            var parsedValues = row.Values
                                .Select(v => columnParser(v))
                                .ToArray();

                            Array.Copy(parsedValues, 0, data,
                                copyIndexStart, sampleSize);

                            copyIndexStart += sampleSize;
                        }
                        currentIndex++;
                    }

                    return new DataBatch<T>(data, sampleShape, batchSampleCount);
                }

                return new DataLoader<T>(LoadCsvData, totalSampleCount);
            }
        }
    }
}
