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
        /// <param name="dimX"></param>
        /// <param name="dimY"></param>
        public RowMajorTensorIndexer2D(Tensor<T> tensor, int dimX, int dimY)
        {
            if (tensor == null)
            { throw new ArgumentNullException(nameof(tensor)); }

            Shape = new TensorShape(dimX, dimY);
            if (Shape.NumberOfElements != tensor.NumberOfElements)
            {
                throw new ArgumentException($"Indexer number of elements: {Shape.NumberOfElements} does not match tensor number of elements: {tensor.NumberOfElements}");
            }

            m_tensor = tensor;
            DimXCount = dimX;
            DimYCount = dimY;
        }

        /// <summary>
        /// 
        /// </summary>
        public int DimXCount { get; }

        /// <summary>
        /// 
        /// </summary>
        public int DimYCount { get; }

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
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public T At(int x, int y)
        {
            var index = x * DimYCount + y;
            return m_tensor.Data[index];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="value"></param>
        public void At(int x, int y, T value)
        {
            var index = x * DimYCount + y;
            m_tensor.Data[index] = value;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="y"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeX(int y, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(y, i);
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeY(int x, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(i, x);
            }
        }
    }
}
