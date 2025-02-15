using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.CsvRowTransforms;

/// <summary>
/// Transforms the selected features with multiple values into a set of binary features.
/// This is refereed to as one-hot encoding:
/// https://en.wikipedia.org/wiki/One-hot
/// For example day { monday, tuesday } becomes
/// day_monday { 1, 0}
/// day_tuesday {0, 1}
/// </summary>
[Serializable]
public sealed class OneHotTransformer : ICsvRowTransformer
{
    readonly Dictionary<string, HashSet<string>> m_featureMap;
    readonly string[] m_columnsToMap;

    /// <summary>
    /// Transforms the selected features with multiple values into a set of binary features.
    /// This is refereed to as one-hot encoding:
    /// https://en.wikipedia.org/wiki/One-hot
    /// For example day { monday, tuesday } becomes
    /// day_monday { 1, 0}
    /// day_tuesday {0, 1}
    /// </summary>
    public OneHotTransformer(params string[] columnsToMap)
    {
        m_featureMap = [];
        m_columnsToMap = columnsToMap;
    }

    /// <summary>
    /// Transforms the selected features with multiple values into a set of binary features.
    /// This is refereed to as one-hot encoding:
    /// https://en.wikipedia.org/wiki/One-hot
    /// For example day { monday, tuesday } becomes
    /// day_monday { 1, 0}
    /// day_tuesday {0, 1}
    /// </summary>
    /// <param name="rows"></param>
    /// <returns></returns>
    public IEnumerable<CsvRow> Transform(IEnumerable<CsvRow> rows)
    {
        if (m_featureMap.Count == 0)
        {
            foreach (var column in m_columnsToMap)
            {
                m_featureMap.Add(column, []);
            }
        }

        // build map
        BuildFeatureMap(rows, m_columnsToMap);

        // add encoded features
        var newColumnNameToIndex = NewColumnNameToIndex(rows);
        var additionalFeatures = m_featureMap.Sum(v => v.Value.Count);

        foreach (var row in rows)
        {
            var newValues = Enumerable.Range(0, row.Values.Length + additionalFeatures)
                .Select(v => "0").ToArray();

            row.Values.CopyTo(newValues, 0);
            foreach (var column in m_columnsToMap)
            {
                var value = row.GetValue(column).Trim();
                var key = column + "_" + value;
                newValues[newColumnNameToIndex[key]] = "1";
            }

            yield return new CsvRow(newColumnNameToIndex, newValues);
        }
    }

    Dictionary<string, int> NewColumnNameToIndex(IEnumerable<CsvRow> rows)
    {
        var newColumnNameToIndex = rows.First().ColumnNameToIndex.ToDictionary(v => v.Key, v => v.Value);
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
                var value = row.GetValue(column).Trim();
                if (!m_featureMap[column].Contains(value))
                {
                    m_featureMap[column].Add(value);
                }
            }
        }
    }
}
