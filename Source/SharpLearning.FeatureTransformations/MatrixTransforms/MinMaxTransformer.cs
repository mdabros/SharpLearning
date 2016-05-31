using SharpLearning.Containers.Matrices;
using SharpLearning.FeatureTransformations.Normalization;
using System;
using System.Collections.Generic;

namespace SharpLearning.FeatureTransformations.MatrixTransforms
{
    /// <summary>
    /// Normalizes features within the specified range (min, max)
    /// </summary>
    public sealed class MinMaxTransformer : IF64MatrixTransform
    {
        class FeatureMinMax
        {
            public double Min {get; set;}
            public double Max { get; set; }
        }

        readonly double m_min;
        readonly double m_max;
        readonly LinearNormalizer m_normalizer = new LinearNormalizer();

        Dictionary<int, FeatureMinMax> m_featureMinMax;

        /// <summary>
        /// Normalizes features within the specified range (min, max)
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        public MinMaxTransformer(double min, double max)
        {
            if (max <= min) { throw new ArgumentException("Max: " + max + "must be larger than Min: " + min); }
            m_featureMinMax = new Dictionary<int, FeatureMinMax>();

            m_min = min;
            m_max = max;
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
                    m_featureMinMax.Add(i, new FeatureMinMax { Min = double.MaxValue, Max = double.MinValue });
                }
            }

            CreateFeatureMinMax(matrix);

            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    var value = matrix[j, i];
                    var minMax = m_featureMinMax[i];
                    var newValue = m_normalizer.Normalize(m_min, m_max, minMax.Min, minMax.Max, value);
                    output[j, i] = newValue;
                }
            }        
        }

        void CreateFeatureMinMax(F64Matrix matrix)
        {
            var rows = matrix.GetNumberOfRows();
            var cols = matrix.GetNumberOfColumns();

            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    var minMax = m_featureMinMax[i];
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
