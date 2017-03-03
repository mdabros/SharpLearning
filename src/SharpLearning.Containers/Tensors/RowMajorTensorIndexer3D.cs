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
        /// <param name="dimX"></param>
        /// <param name="dimY"></param>
        /// <param name="dimZ"></param>
        public RowMajorTensorIndexer3D(Tensor<T> tensor, int dimX, int dimY, int dimZ)
        {
            if (tensor == null)
            { throw new ArgumentNullException(nameof(tensor)); }

            Shape = new TensorShape(dimX, dimY, dimZ);
            if (Shape.NumberOfElements != tensor.NumberOfElements)
            {
                throw new ArgumentException($"Indexer number of elements: {Shape.NumberOfElements} does not match tensor number of elements: {tensor.NumberOfElements}");
            }

            m_tensor = tensor;
            DimXCount = dimX;
            DimYCount = dimY;
            DimZCount = dimZ;
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
        /// <returns></returns>
        public T At(int x, int y, int z)
        {
            var index = ((this.DimXCount * y) + x) * this.DimZCount + z;
            return m_tensor.Data[index];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <param name="value"></param>
        public void At(int x, int y, int z, T value)
        {
            var index = ((this.DimXCount * y) + x) * this.DimZCount + z;
            m_tensor.Data[index] = value;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeX(int y, int z, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(i, y, z);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="z"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeY(int x, int z, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(x, i, z);
            }
        }
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeZ(int x, int y, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(x, y, i);
            }
        }
    }
}
