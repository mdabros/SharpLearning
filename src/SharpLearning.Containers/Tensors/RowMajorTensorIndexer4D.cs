using SharpLearning.Containers.Views;
using System;
using System.Diagnostics;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class RowMajorTensorIndexer4D<T> : ITensorIndexer4D<T>
    {
        Tensor<T> m_tensor;


        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="depth"></param>
        /// <param name="dim4"></param>
        public RowMajorTensorIndexer4D(Tensor<T> tensor, int height, int width, int depth, int dim4)
        {
            if (tensor == null)
            { throw new ArgumentNullException(nameof(tensor)); }

            Shape = new TensorShape(height, width, depth, dim4);
            if (Shape.NumberOfElements != tensor.NumberOfElements)
            {
                throw new ArgumentException($"Indexer number of elements: {Shape.NumberOfElements} does not match tensor number of elements: {tensor.NumberOfElements}");
            }

            m_tensor = tensor;
            Height = height;
            Width = width;
            Depth = depth;
            Dim4 = dim4;
        }

        /// <summary>
        /// 
        /// </summary>
        public int Height { get; }

        /// <summary>
        /// 
        /// </summary>
        public int Width { get; }

        /// <summary>
        /// 
        /// </summary>
        public int Depth { get; }

        /// <summary>
        /// 
        /// </summary>
        public int Dim4 { get; }

        /// <summary>
        /// 
        /// </summary>
        public TensorShape Shape { get; }

        /// <summary>
        /// 
        /// </summary>
        public int NumberOfElements { get { return Shape.NumberOfElements; } }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="d"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public T At(int w, int h, int d, int n)
        {
            var index = ((n * Depth + d) * Width + h) * Height + w;
            return m_tensor.Data[index];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="d"></param>
        /// <param name="n"></param>
        /// <param name="value"></param>
        public void At(int w, int h, int d, int n, T value)
        {
            //var index = x + y * DimXCount + z * DimXCount * DimYCount + n * DimXCount * DimYCount * DimZCount;
            var index = ((n * Depth + d) * Width + h) * Height + w;
            m_tensor.Data[index] = value;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="h"></param>
        /// <param name="d"></param>
        /// <param name="n"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeWidth(int h, int d, int n, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(i, h, d, n);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="d"></param>
        /// <param name="n"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeHeight(int w, int d, int n, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(w, i, d, n);
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="n"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeDepth(int w, int h, int n, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(w, h, i, n);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="c"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeDim4(int w, int h, int c, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(w, h, c, i);
            }
        }
    }
}
