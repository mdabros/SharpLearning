using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.FeatureTransformations.Normalization
{
    /// <summary>
    /// Normalizes features within the specified range (min, max)
    /// </summary>
    public sealed class FeatureNormalizationTransformer
    {
        class FeatureMinMax
        {
            public double Min {get; set;}
            public double Max { get; set; }
        }

        readonly double m_min;
        readonly double m_max;
        readonly LinearNormalizer m_normalizer = new LinearNormalizer();

        Dictionary<string, FeatureMinMax> m_featureMinMax;

        /// <summary>
        /// Normalizes features within the specified range (min, max)
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        public FeatureNormalizationTransformer(double min, double max)
        {
            if (max <= min) { throw new ArgumentException("Max: " + max + "must be larger than Min: " + min); }
            m_featureMinMax = new Dictionary<string, FeatureMinMax>();

            m_min = min;
            m_max = max;
        }

        /// <summary>
        /// Normalizes features within the specified range (min, max)
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="dateTimeColumn"></param>
        /// <param name="columnNames"></param>
        /// <returns></returns>
        public IEnumerable<CsvRow> Transform(IEnumerable<CsvRow> rows, params string[] columnNames)
        {
            if(m_featureMinMax.Count == 0)
            {
                foreach (var name in columnNames)
                {
                    m_featureMinMax.Add(name, new FeatureMinMax { Min = double.MaxValue, Max = double.MinValue });
                }
            }

            CreateFeatureMinMax(rows, columnNames);

            foreach (var row in rows)
            {
                foreach (var name in columnNames)
                {
                    var value = FloatingPointConversion.ToF64(row.GetValue(name));
                    var minMax = m_featureMinMax[name];
                    var newValue = m_normalizer.Normalize(m_min, m_max, minMax.Min, minMax.Max, value);
                    row.SetValue(name, FloatingPointConversion.ToString(newValue));
                }

                yield return row;
            }
        }

        void CreateFeatureMinMax(IEnumerable<CsvRow> rows, string[] columnNames)
        {
            foreach (var row in rows)
            {
                foreach (var name in columnNames)
                {
                    var minMax = m_featureMinMax[name];
                    var value = FloatingPointConversion.ToF64(row.GetValue(name));
                    if (value < minMax.Min)
                    {
                        minMax.Min = value;
                    }
                    else if (value > minMax.Max)
                    {
                        minMax.Max = value;
                    }
                }
            }
        }
    }
}
