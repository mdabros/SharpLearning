using SharpLearning.InputOutput.Csv;
using System.Collections.Generic;

namespace SharpLearning.FeatureTransformations
{
    /// <summary>
    /// Maps categorical features to forth running integer values. 
    /// This is usefull for transforming features containing strings into numerical categories.
    /// For example: [monday, tuesday] -> [0, 1]
    /// This is needed when the features are used with machine learning algorithms
    /// </summary>
    public sealed class MapCategoricalFeaturesTransformer
    {
        readonly Dictionary<string, Dictionary<string, string>> m_namedFeatureMapping;
        readonly Dictionary<string, int> m_namedCategoricalMapping;

        /// <summary>
        /// 
        /// </summary>
        public MapCategoricalFeaturesTransformer()
        {
            m_namedFeatureMapping = new Dictionary<string, Dictionary<string, string>>();
            m_namedCategoricalMapping = new Dictionary<string, int>();
        }

        /// <summary>
        /// Maps categorical features to forth running integer values. 
        /// This is usefull for transforming features containing strings into numerical categories.
        /// For example: [monday, tuesday] -> [0, 1]
        /// This is needed when the features are used with machine learning algorithms
        /// </summary>
        /// <param name="rows">the rows to be transformed</param>
        /// <param name="columnsToMap">the name of the csv columns to map</param>
        /// <returns></returns>
        public IEnumerable<CsvRow> Transform(IEnumerable<CsvRow> rows, params string[] columnsToMap)
        {
            if(m_namedFeatureMapping.Count == 0)
            {
                foreach (var column in columnsToMap)
                {
                    m_namedFeatureMapping.Add(column, new Dictionary<string, string>());
                    m_namedCategoricalMapping.Add(column, 0);
                }
            }

            foreach (var row in rows)
            {
                foreach (var column in columnsToMap)
                {
                    var columnMap = m_namedFeatureMapping[column];
                    var value = row.GetValue(column);
                    if(columnMap.ContainsKey(value))
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
}
