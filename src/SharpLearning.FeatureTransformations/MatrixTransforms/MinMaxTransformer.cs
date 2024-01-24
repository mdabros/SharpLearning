using System;
using System.Collections.Generic;
using SharpLearning.Containers.Matrices;
using SharpLearning.FeatureTransformations.Normalization;

namespace SharpLearning.FeatureTransformations.MatrixTransforms
{
    /// <summary>
    /// Normalizes features within the specified range (min, max)
    /// </summary>
    [Serializable]
    public sealed class MinMaxTransformer : IF64MatrixTransform, IF64VectorTransform
    {
        [Serializable]
        class FeatureMinMax
        {
            public double Min { get; set; }
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
            var rows = matrix.RowCount;
            var cols = matrix.ColumnCount;
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
            var rows = matrix.RowCount;
            var cols = matrix.ColumnCount;

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
            var rows = matrix.RowCount;
            var cols = matrix.ColumnCount;

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

        /// <summary>
        /// Normalizes features within the specified range (min, max)
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public double[] Transform(double[] vector)
        {
            var output = new double[vector.Length];
            Transform(vector, output);

            return output;
        }

        /// <summary>
        /// Normalizes features within the specified range (min, max)
        /// </summary>
        /// <param name="vector"></param>
        /// <param name="output"></param>
        public void Transform(double[] vector, double[] output)
        {
            if (m_featureMinMax.Count == 0)
            {
                throw new ArgumentException("No feature min max calculated. " +
                    "Feature min max must be calculated from a full matrix before the vector transform can be used");
            }

            var cols = vector.Length;

            for (int i = 0; i < cols; i++)
            {
                var value = vector[i];
                var minMax = m_featureMinMax[i];
                var newValue = m_normalizer.Normalize(m_min, m_max, minMax.Min, minMax.Max, value);
                output[i] = newValue;
            }
        }
    }
}
