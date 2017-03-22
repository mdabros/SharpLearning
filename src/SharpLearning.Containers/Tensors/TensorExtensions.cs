using SharpLearning.Containers.Extensions;
using System;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public static class TensorExtensions
    {

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor"></param>
        /// <param name="func"></param>
        public static void Map<T>(this Tensor<T> tensor, Func<T> func)
        {
            tensor.Data.Map(func);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="func"></param>
        public static void Map<T>(this Tensor<T> tensor, Func<T, T> func)
        {
            tensor.Data.Map(func);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="output"></param>
        public static void Subtract(this Tensor<float> t1, Tensor<float> t2, Tensor<float> output)
        {
            if(t1.ElementCount != t2.ElementCount)
            { throw new ArgumentException($"t1 element count: {t1.ElementCount} differs from t2: {t2.ElementCount}"); }

            var t1Data = t1.Data;
            var t2Data = t2.Data;
            var tOutData = output.Data;

            for (int i = 0; i < t1.ElementCount; i++)
            {
                tOutData[i] = t1Data[i] - t2Data[i];
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="output"></param>
        public static void Subtract(this Tensor<double> t1, Tensor<double> t2, Tensor<double> output)
        {
            if (t1.ElementCount != t2.ElementCount)
            { throw new ArgumentException($"t1 element count: {t1.ElementCount} differs from t2: {t2.ElementCount}"); }

            var t1Data = t1.Data;
            var t2Data = t2.Data;
            var tOutData = output.Data;

            for (int i = 0; i < t1.ElementCount; i++)
            {
                tOutData[i] = t1Data[i] - t2Data[i];
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="output"></param>
        public static void PointwiseMultiply(this Tensor<float> t1, Tensor<float> t2, Tensor<float> output)
        {
            if (t1.ElementCount != t2.ElementCount)
            { throw new ArgumentException($"t1 element count: {t1.ElementCount} differs from t2: {t2.ElementCount}"); }

            var t1Data = t1.Data;
            var t2Data = t2.Data;
            var tOutData = output.Data;

            for (int i = 0; i < t1.ElementCount; i++)
            {
                tOutData[i] = t1Data[i] * t2Data[i];
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="output"></param>
        public static void PointwiseMultiply(this Tensor<double> t1, Tensor<double> t2, Tensor<double> output)
        {
            if (t1.ElementCount != t2.ElementCount)
            { throw new ArgumentException($"t1 element count: {t1.ElementCount} differs from t2: {t2.ElementCount}"); }

            var t1Data = t1.Data;
            var t2Data = t2.Data;
            var tOutData = output.Data;

            for (int i = 0; i < t1.ElementCount; i++)
            {
                tOutData[i] = t1Data[i] * t2Data[i];
            }
        }

        /// <summary>
        /// Adds vector v to matrix m. V is Added to each row of the matrix
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void AddRowWise(this Tensor<float> m, float[] v, Tensor<float> output)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];

            var mData = m.Data;
            var outData = output.Data;
            var vData = v;

            if (v.Length != cols)
            { throw new ArgumentException("matrix cols: " + cols + " differs from vector length: " + v.Length); }

            for (int row = 0; row < rows; row++)
            {
                var rowOffSet = cols * row;
                for (int col = 0; col < cols; col++)
                {
                    var mIndex = rowOffSet + col;
                    outData[mIndex] = mData[mIndex] + vData[col];
                }
            }
        }

        /// <summary>
        /// Adds vector v to matrix m. V is Added to each row of the matrix
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void AddRowWise(this Tensor<double> m, double[] v, Tensor<double> output)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];

            var mData = m.Data;
            var outData = output.Data;
            var vData = v;

            if (v.Length != cols)
            { throw new ArgumentException("matrix cols: " + cols + " differs from vector length: " + v.Length); }

            for (int row = 0; row < rows; row++)
            {
                var rowOffSet = cols * row;
                for (int col = 0; col < cols; col++)
                {
                    var mIndex = rowOffSet + col;
                    outData[mIndex] = mData[mIndex] + vData[col];
                }
            }
        }

        /// <summary>
        /// Sums the columns of m into the vector sums. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="sums"></param>
        public static void SumColumns(this Tensor<float> m, float[] sums)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];
            var mData = m.Data;

            for (int col = 0; col < cols; col++)
            {
                for (int row = 0; row < rows; row++)
                {
                    var rowOffSet = row * cols;
                    var mIndex = rowOffSet + col;
                    sums[col] += mData[mIndex];
                }
            }
        }

        /// <summary>
        /// Sums the columns of m into the vector sums. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="sums"></param>
        public static void SumColumns(this Tensor<double> m, double[] sums)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];
            var mData = m.Data;

            for (int col = 0; col < cols; col++)
            {
                for (int row = 0; row < rows; row++)
                {
                    var rowOffSet = row * cols;
                    var mIndex = rowOffSet + col;
                    sums[col] += mData[mIndex];
                }
            }
        }
    }
}
