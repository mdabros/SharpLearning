using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.FeatureTransformations
{
    /// <summary>
    /// Sums column value by column value.
    /// Each unique value in the supplied sumByColumnName column
    /// is used to sum the occurences of the supplied valueToSum across all other selected columns
    /// </summary>
    public sealed class SumColumnValueByColumnValueTransformer
    {
        static int Count = 0;

        readonly Dictionary<string, double> m_columnValueToCount;
        readonly string m_countByColumnName;
        readonly string m_valueToCount;

        /// <summary>
        /// Sums column value by column value.
        /// Each unique value in the supplied sumByColumnName column
        /// is used to sum the occurences of the supplied valueToSum across all other selected columns
        /// </summary>
        public SumColumnValueByColumnValueTransformer(string sumByColumnName, string valueToSum)
        {
            m_countByColumnName = sumByColumnName;
            m_valueToCount = valueToSum;
            m_columnValueToCount = new Dictionary<string, double>();
        }

        /// <summary>
        /// Sums column value by column value.
        /// Each unique value in the supplied sumByColumnName column
        /// is used to sum the occurences of the supplied valueToSum across all other selected columns
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="columnsToSearch"></param>
        /// <returns></returns>
        public IEnumerable<CsvRow> Transform(IEnumerable<CsvRow> rows, params string[] columnsToSearch)
        {
            if(columnsToSearch.Length == 0)
            {
                columnsToSearch = rows.First().ColumnNameToIndex.Keys.Except(new string[] { m_countByColumnName }).ToArray();
            }

            if (m_columnValueToCount.Count == 0)
            {
                foreach (var row in rows)
                {
                    var countColumnValue = row.GetValue(m_countByColumnName);
                    if (!m_columnValueToCount.ContainsKey(countColumnValue))
                    {
                        m_columnValueToCount.Add(countColumnValue, 0.0);
                    }

                    var sum = row.GetValues(columnsToSearch)
                        .Where(v => v == m_valueToCount)
                        .Count();

                    m_columnValueToCount[countColumnValue] += sum;

                    Trace.WriteLine(Count++);
                }
            }

            var newColumnNameToIndex = rows.First().ColumnNameToIndex.ToDictionary(v => v.Key, v => v.Value);
            var newColumnName = m_countByColumnName + "_" + m_valueToCount + "_Counts";
            newColumnNameToIndex.Add(newColumnName, newColumnNameToIndex.Count);

            foreach (var row in rows)
            {
                var newValues = new string[row.Values.Length + 1];
                row.Values.CopyTo(newValues, 0);
                newValues[row.Values.Length] = FloatingPointConversion
                    .ToString(m_columnValueToCount[row.GetValue(m_countByColumnName)]);

                yield return new CsvRow(newColumnNameToIndex, newValues);
            }
        }
    }
}
