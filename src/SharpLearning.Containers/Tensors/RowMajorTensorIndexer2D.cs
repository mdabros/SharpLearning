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
        /// <param name="width"></param>
        /// <param name="height"></param>
        public RowMajorTensorIndexer2D(Tensor<T> tensor, int width, int height)
        {
            if (tensor == null)
            { throw new ArgumentNullException(nameof(tensor)); }

            Shape = new TensorShape(width, height);
            if (Shape.NumberOfElements != tensor.NumberOfElements)
            {
                throw new ArgumentException($"Indexer number of elements: {Shape.NumberOfElements} does not match tensor number of elements: {tensor.NumberOfElements}");
            }

            m_tensor = tensor;
            Width = width;
            Height = height;
        }

        /// <summary>
        /// 
        /// </summary>
        public int Width { get; }

        /// <summary>
        /// 
        /// </summary>
        public int Height { get; }

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
        /// <returns></returns>
        public T At(int w, int h)
        {
            var index = w * Height + h;
            return m_tensor.Data[index];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="value"></param>
        public void At(int w, int h, T value)
        {
            var index = w * Height + h;
            m_tensor.Data[index] = value;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="h"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeWidth(int h, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(h, i);
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void Rangeheight(int w, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(i, w);
            }
        }
    }
}
