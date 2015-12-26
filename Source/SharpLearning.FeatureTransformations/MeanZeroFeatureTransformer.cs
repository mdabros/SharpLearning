using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.FeatureTransformations
{
    /// <summary>
    /// Shift data so the mean of each of the selected columns is zero
    /// </summary>
    public sealed class MeanZeroFeatureTransformer
    {
        readonly Dictionary<string, double> m_featureMean;

        public MeanZeroFeatureTransformer()
        {
            m_featureMean = new Dictionary<string, double>();
        }

        /// <summary>
        /// Shift data so the mean of each of the selected columns is zero
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="dateTimeColumn"></param>
        /// <param name="columnNames"></param>
        /// <returns></returns>
        public IEnumerable<CsvRow> Transform(IEnumerable<CsvRow> rows, params string[] columnNames)
        {
            if (m_featureMean.Count == 0)
            {
                foreach (var name in columnNames)
                {
                    m_featureMean.Add(name, 0.0);
                }

                var rowCount = 0;
                foreach (var row in rows)
                {
                    foreach (var name in columnNames)
                    {
                        m_featureMean[name] += FloatingPointConversion.ToF64(row.GetValue(name));
                    }
                    rowCount++;
                }

                foreach (var name in columnNames)
                {
                    m_featureMean[name] /= (double)rowCount;
                }
            }


            foreach (var row in rows)
            {
                foreach (var name in columnNames)
                {
                    var value = FloatingPointConversion.ToF64(row.GetValue(name));
                    var mean = m_featureMean[name];
                    var newValue = value - mean;
                    row.SetValue(name, FloatingPointConversion.ToString(newValue));
                }

                yield return row;
            }
        }
    }
}
