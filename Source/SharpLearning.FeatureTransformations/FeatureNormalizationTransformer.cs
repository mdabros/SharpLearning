using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.FeatureTransformations.Normalization;
using SharpLearning.InputOutput.Csv;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.FeatureTransformations
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

        /// <summary>
        /// Normalizes features within the specified range (min, max)
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public F64Matrix Transform(F64Matrix matrix)
        {
            var rows = matrix.GetNumberOfRows();
            var cols = matrix.GetNumberOfColumns();
            var output = new F64Matrix(rows, cols);

            Transform(matrix, output);

            return output;
        }

        /// <summary>
        /// Normalizes features within the specified range (min, max)
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="output"></param>
        public void Transform(F64Matrix matrix, F64Matrix output)
        {
            var rows = matrix.GetNumberOfRows();
            var cols = matrix.GetNumberOfColumns();

            if (m_featureMinMax.Count == 0)
            {
                for (int i = 0; i < cols; i++)
                {
                    m_featureMinMax.Add(i.ToString(), new FeatureMinMax { Min = double.MaxValue, Max = double.MinValue });
                }
            }

            CreateFeatureMinMax(matrix);

            for (int i = 0; i < cols; i++)
            {
                var name = i.ToString();
                for (int j = 0; j < rows; j++)
                {
                    var value = matrix[j, i];
                    var minMax = m_featureMinMax[name];
                    var newValue = m_normalizer.Normalize(m_min, m_max, minMax.Min, minMax.Max, value);
                    output[j, i] = newValue;
                }
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

        void CreateFeatureMinMax(F64Matrix matrix)
        {
            var rows = matrix.GetNumberOfRows();
            var cols = matrix.GetNumberOfColumns();

            for (int i = 0; i < cols; i++)
            {
                var name = i.ToString();
                for (int j = 0; j < rows; j++)
                {
                    var minMax = m_featureMinMax[name];
                    var value = matrix[j, i];
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
