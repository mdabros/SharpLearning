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
        /// <param name="dimX"></param>
        /// <param name="dimY"></param>
        /// <param name="dimZ"></param>
        /// <param name="dimH"></param>
        public RowMajorTensorIndexer4D(Tensor<T> tensor, int dimX, int dimY, int dimZ, int dimH)
        {
            if (tensor == null)
            { throw new ArgumentNullException(nameof(tensor)); }

            Shape = new TensorShape(dimX, dimY, dimZ, dimH);
            if (Shape.NumberOfElements != tensor.NumberOfElements)
            {
                throw new ArgumentException($"Indexer number of elements: {Shape.NumberOfElements} does not match tensor number of elements: {tensor.NumberOfElements}");
            }

            m_tensor = tensor;
            DimXCount = dimX;
            DimYCount = dimY;
            DimZCount = dimZ;
            DimNCount = dimH;
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
        public int DimZCount { get; }

        /// <summary>
        /// 
        /// </summary>
        public int DimNCount { get; }

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
        /// <param name="z"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public T At(int x, int y, int z, int n)
        {
            var index = ((n * DimZCount + z) * DimYCount + y) * DimXCount + x;
            return m_tensor.Data[index];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <param name="n"></param>
        /// <param name="value"></param>
        public void At(int x, int y, int z, int n, T value)
        {
            //var index = x + y * DimXCount + z * DimXCount * DimYCount + n * DimXCount * DimYCount * DimZCount;
            var index = ((n * DimZCount + z) * DimYCount + y) * DimXCount + x;
            m_tensor.Data[index] = value;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <param name="n"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeX(int y, int z, int n, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(i, y, z, n);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="z"></param>
        /// <param name="n"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeY(int x, int z, int n, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(x, i, z, n);
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="n"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeZ(int x, int y, int n, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(x, y, i, n);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeN(int x, int y, int z, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(x, y, z, i);
            }
        }
    }
}
