using SharpLearning.InputOutput.Csv;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.FeatureTransformations
{
    /// <summary>
    /// Transforms the selected features with multiple values into a set of binary features.
    /// This is refered to as one-hot encoding:
    /// https://en.wikipedia.org/wiki/One-hot
    /// Forexample day { monday, tuesday } becomes
    /// day_monday { 1, 0}
    /// day_tuesday {0, 1}
    /// </summary>
    public sealed class OneHotTransformer
    {
        readonly Dictionary<string, HashSet<string>> m_featureMap;

        /// <summary>
        /// Transforms the selected features with multiple values into a set of binary features.
        /// This is refered to as one-hot encoding:
        /// https://en.wikipedia.org/wiki/One-hot
        /// Forexample day { monday, tuesday } becomes
        /// day_monday { 1, 0}
        /// day_tuesday {0, 1}
        /// </summary>
        public OneHotTransformer()
        {
            m_featureMap = new Dictionary<string, HashSet<string>>();
        }

        /// <summary>
        /// Transforms the selected features with multiple values into a set of binary features.
        /// This is refered to as one-hot encoding:
        /// https://en.wikipedia.org/wiki/One-hot
        /// Forexample day { monday, tuesday } becomes
        /// day_monday { 1, 0}
        /// day_tuesday {0, 1}
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="columnsToMap"></param>
        /// <returns></returns>
        public IEnumerable<CsvRow> Transform(IEnumerable<CsvRow> rows, params string[] columnsToMap)
        {
            if(m_featureMap.Count == 0)
            {
                foreach (var column in columnsToMap)
                {
                    m_featureMap.Add(column, new HashSet<string>());
                }
            }

            // build map
            BuildFeatureMap(rows, columnsToMap);

            // add encoded fetures
            var newColumnNameToIndex = NewColumnNameToIndex(rows);
            var additionalFeatures = m_featureMap.Select(v => v.Value.Count).Sum();

            foreach (var row in rows)
            {
                var newValues = Enumerable.Range(0, row.Values.Length + additionalFeatures).Select(v => "0").ToArray();

                row.Values.CopyTo(newValues, 0);
                foreach (var column in columnsToMap)
                {
                    var value = row.GetValue(column);
                    var key = column + "_" + value;
                    newValues[newColumnNameToIndex[key]] = "1";
                }

                yield return new CsvRow(newColumnNameToIndex, newValues);
            }
        }

        Dictionary<string, int> NewColumnNameToIndex(IEnumerable<CsvRow> rows)
        {
            var newColumnNameToIndex = rows.First().ColumnNameToIndex;
            var index = newColumnNameToIndex.Count;

            foreach (var columnSet in m_featureMap)
            {
                var name = columnSet.Key;
                var values = columnSet.Value;

                foreach (var value in values)
                {
                    newColumnNameToIndex.Add(name + "_" + value, index++);
                }
            }
            return newColumnNameToIndex;
        }

        void BuildFeatureMap(IEnumerable<CsvRow> rows, string[] columnsToMap)
        {
            foreach (var row in rows)
            {
                foreach (var column in columnsToMap)
                {
                    var value = row.GetValue(column);
                    if (!m_featureMap[column].Contains(value))
                    {
                        m_featureMap[column].Add(value);
                    }
                }
            }
        }
    }
}
