using SharpLearning.Containers.Views;
using System;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class RowMajorTensorIndexer2D<T> 
        : ITensorIndexer2D<T>
        , IEquatable<RowMajorTensorIndexer2D<T>>
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
            if (Shape.ElementCount != tensor.ElementCount)
            {
                throw new ArgumentException($"Indexer number of elements: {Shape.ElementCount} does not match tensor number of elements: {tensor.ElementCount}");
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
        public int NumberOfElements { get { return Shape.ElementCount; } }


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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(RowMajorTensorIndexer2D<T> other)
        {
            if (!this.m_tensor.Equals(other.m_tensor)) { return false; }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            RowMajorTensorIndexer2D<T> other = obj as RowMajorTensorIndexer2D<T>;
            if (other != null && Equals(other))
            {
                return true;
            }

            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                int hash = 17;
                hash = hash * 23 + m_tensor.GetHashCode();
                hash = hash * 23 + Shape.GetHashCode();

                return hash;
            }
        }
    }
}
