using SharpLearning.Containers.Views;
using System;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class RowMajorTensorIndexer2D<T> : ITensorIndexer2D<T>
    {
        Tensor<T> m_tensor;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        public RowMajorTensorIndexer2D(Tensor<T> tensor, int h, int w)
        {
            if (tensor == null)
            { throw new ArgumentNullException(nameof(tensor)); }

            Shape = new TensorShape(h, w);
            if (Shape.NumberOfElements != tensor.NumberOfElements)
            {
                throw new ArgumentException($"Indexer number of elements: {Shape.NumberOfElements} does not match tensor number of elements: {tensor.NumberOfElements}");
            }

            m_tensor = tensor;
            H = h;
            W = w;
        }

        /// <summary>
        /// 
        /// </summary>
        public int H { get; }

        /// <summary>
        /// 
        /// </summary>
        public int W { get; }

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
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        public T At(int h, int w)
        {
            var index = h * W + w;
            return m_tensor.Data[index];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="value"></param>
        public void At(int h, int w, T value)
        {
            var index = h * W + w;
            m_tensor.Data[index] = value;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeH(int w, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(i, w);
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="h"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeW(int h, Interval1D interval, T[] output)
        {
            // row-major makes direct copy of rows possible.
            var startIndex = h * W + interval.FromInclusive;
            Array.Copy(m_tensor.Data, startIndex, output, 0, interval.ToExclusive);
        }
    }
}
