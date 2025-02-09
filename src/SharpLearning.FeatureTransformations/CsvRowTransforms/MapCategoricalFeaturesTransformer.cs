using System;
using System.Collections.Generic;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.CsvRowTransforms;

/// <summary>
/// Maps categorical features to forth running integer values. 
/// This is usefull for transforming features containing strings into numerical categories.
/// For example: [monday, tuesday] -> [0, 1]
/// This is needed when the features are used with machine learning algorithms
/// </summary>
[Serializable]
public sealed class MapCategoricalFeaturesTransformer : ICsvRowTransformer
{
    readonly Dictionary<string, Dictionary<string, string>> m_namedFeatureMapping;
    readonly Dictionary<string, int> m_namedCategoricalMapping;
    readonly string[] m_columnsToMap;

    /// <summary>
    /// 
    /// </summary>
    public MapCategoricalFeaturesTransformer(params string[] columnsToMap)
    {
        m_namedFeatureMapping = [];
        m_namedCategoricalMapping = [];
        m_columnsToMap = columnsToMap;
    }

    /// <summary>
    /// Maps categorical features to forth running integer values. 
    /// This is usefull for transforming features containing strings into numerical categories.
    /// For example: [monday, tuesday] -> [0, 1]
    /// This is needed when the features are used with machine learning algorithms
    /// </summary>
    /// <param name="rows">the rows to be transformed</param>
    /// <returns></returns>
    public IEnumerable<CsvRow> Transform(IEnumerable<CsvRow> rows)
    {
        if (m_namedFeatureMapping.Count == 0)
        {
            foreach (var column in m_columnsToMap)
            {
                m_namedFeatureMapping.Add(column, []);
                m_namedCategoricalMapping.Add(column, 0);
            }
        }

        foreach (var row in rows)
        {
            foreach (var column in m_columnsToMap)
            {
                var columnMap = m_namedFeatureMapping[column];
                var value = row.GetValue(column);
                if (columnMap.ContainsKey(value))
                {
                    row.SetValue(column, columnMap[value]);
                }
                else
                {
                    columnMap[value] = m_namedCategoricalMapping[column]++.ToString();
                    row.SetValue(column, columnMap[value]);
                }
            }

            yield return row;
        }
    }
}
