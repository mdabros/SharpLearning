using System;
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
        /// Creates DataLoader from Csv Text.
        /// Note that the sample shape is inferred from the number of column names.
        /// So rank 1 is assumed.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="csvTextData"></param>
        /// <param name="columnParser"></param>
        /// <param name="columnNames"></param>
        /// <returns></returns>
        public static DataLoader<T> FromCsvText<T>(string csvTextData,
            Func<string, T> columnParser,
            params string[] columnNames) => FromCsv<T>(CsvParser.FromText(csvTextData),
                columnParser, columnNames, new int[] { columnNames.Length });

        /// <summary>
        /// Creates DataLoader from Csv file.
        /// Note that the sample shape is inferred from the number of column names.
        /// So rank 1 is assumed.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="filePath"></param>
        /// <param name="columnParser"></param>
        /// <param name="columnNames"></param>
        /// <returns></returns>
        public static DataLoader<T> FromCsvFile<T>(string filePath,
            Func<string, T> columnParser,
            params string[] columnNames) => FromCsv<T>(CsvParser.FromFile(filePath),
                columnParser, columnNames, new int[] { columnNames.Length });
        
        /// <summary>
        /// Creates a DataLoader reading from Csv Data.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="parser"></param>
        /// <param name="columnParser"></param>
        /// <param name="columnNames"></param>
        /// <param name="sampleShape"></param>
        /// <returns></returns>
        public static DataLoader<T> FromCsv<T>(CsvParser parser,
            Func<string, T> columnParser,
            string[] columnNames,
            int[] sampleShape)
        {
            var sampleSize = sampleShape.Aggregate((v1, v2) => v1 * v2);

            DataBatch<T> LoadCsvData(int[] indices)
            {
                var rows = parser.EnumerateRows(columnNames);
                var batchSampleCount = indices.Length;
                var data = new T[batchSampleCount * sampleSize];
                var index = 0;
                var startIndex = 0;

                foreach (var row in rows)
                {
                    if(indices.Contains(index))
                    {
                        var transformed = row.Values
                            .Select(v => columnParser(v))
                            .ToArray();

                        Array.Copy(transformed, 0, data, 
                            startIndex, sampleSize);

                        startIndex += sampleSize;
                    }
                    index++;
                }

                return new DataBatch<T>(data, sampleShape, batchSampleCount);
            }

            var totalSampleCount = parser.EnumerateRows(columnNames).Count();
            return new DataLoader<T>(LoadCsvData, totalSampleCount);
        }
    }
}
