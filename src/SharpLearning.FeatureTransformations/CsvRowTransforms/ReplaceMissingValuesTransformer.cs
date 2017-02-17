using SharpLearning.InputOutput.Csv;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.FeatureTransformations.CsvRowTransforms
{
    /// <summary>
    /// Replaces missing values identified with the missing values identifiers. 
    /// The missing values are replaced by the provided replacement value
    /// </summary>
    public sealed class ReplaceMissingValuesTransformer : ICsvRowTransformer
    {
        readonly Dictionary<string, string> m_missingValueIdentifiers;
        readonly string m_replacementValue;

        /// <summary>
        /// Replaces missing values identified with the missing values identifiers. 
        /// The missing values are replaced by the provided replacement value
        /// </summary>
        /// <param name="replacementValue"></param>
        /// <param name="missingValueIdentifiers"></param>
        public ReplaceMissingValuesTransformer(string replacementValue, params string[] missingValueIdentifiers)
        {
            if (missingValueIdentifiers == null) { throw new ArgumentException("missingValueIdentifiers"); }
            if (replacementValue == null) { throw new ArgumentException("replacementValue"); }
            m_missingValueIdentifiers = missingValueIdentifiers.ToDictionary(v => v, v => v);
            m_replacementValue = replacementValue;
        }

        /// <summary>
        /// Replaces missing values identified with the missing values identifiers. 
        /// The missing values are replaced by the provided replacement value
        /// </summary>
        /// <param name="rows"></param>
        /// <returns></returns>
        public IEnumerable<CsvRow> Transform(IEnumerable<CsvRow> rows)
        {
            foreach (var row in rows)
            {
                var values = row.Values;
                for (int i = 0; i < values.Length; i++)
                {
                    var value = values[i];
                    if(m_missingValueIdentifiers.ContainsKey(value))
                    {
                        values[i] = m_replacementValue;
                    }
                }

                yield return row;
            }
        }
    }
}
