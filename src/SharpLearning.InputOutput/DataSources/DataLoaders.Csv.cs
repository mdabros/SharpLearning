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
        /// Creates DataLoader from CsvRows.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="rows"></param>
        /// <param name="columnParser"></param>
        /// <param name="sampleShape"></param>
        /// <returns></returns>
        public static DataLoader<T> ToCsvDataLoader<T>(
            this IEnumerable<IReadOnlyList<string>> rows,
            Func<string, T> columnParser,
            params int[] sampleShape)
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
                        var parsedValues = row
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

            var totalSampleCount = rows.Count();
            return new DataLoader<T>(LoadCsvData, totalSampleCount);
        }
    }
}
