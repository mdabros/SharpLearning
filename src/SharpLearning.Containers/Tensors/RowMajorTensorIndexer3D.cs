using SharpLearning.Containers.Views;
using System;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class RowMajorTensorIndexer3D<T> : ITensorIndexer3D<T>
    {
        Tensor<T> m_tensor;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="depth"></param>
        public RowMajorTensorIndexer3D(Tensor<T> tensor, int width, int height, int depth)
        {
            if (tensor == null)
            { throw new ArgumentNullException(nameof(tensor)); }

            Shape = new TensorShape(width, height, depth);
            if (Shape.NumberOfElements != tensor.NumberOfElements)
            {
                throw new ArgumentException($"Indexer number of elements: {Shape.NumberOfElements} does not match tensor number of elements: {tensor.NumberOfElements}");
            }

            m_tensor = tensor;
            Width = width;
            Height = height;
            Depth = depth;
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
        public int Depth { get; }


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
        /// <returns></returns>
        public T At(int w, int h, int d)
        {
            var index = ((this.Width * h) + w) * this.Depth + d;
            return m_tensor.Data[index];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="d"></param>
        /// <param name="value"></param>
        public void At(int w, int h, int d, T value)
        {
            var index = ((this.Width * h) + w) * this.Depth + d;
            m_tensor.Data[index] = value;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="h"></param>
        /// <param name="d"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeWidth(int h, int d, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(i, h, d);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="d"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeHeight(int w, int d, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(w, i, d);
            }
        }
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeDepth(int h, int w, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(h, w, i);
            }
        }
    }
}
