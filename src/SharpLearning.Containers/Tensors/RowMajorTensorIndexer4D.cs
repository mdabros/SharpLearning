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
        /// <param name="n"></param>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        public RowMajorTensorIndexer4D(Tensor<T> tensor, int n, int c, int h, int w)
        {
            if (tensor == null)
            { throw new ArgumentNullException(nameof(tensor)); }

            Shape = new TensorShape(n, c, h, w);
            if (Shape.ElementCount != tensor.ElementCount)
            {
                throw new ArgumentException($"Indexer number of elements: {Shape.ElementCount} does not match tensor number of elements: {tensor.ElementCount}");
            }

            m_tensor = tensor;
            N = n;
            C = c;
            H = h;
            W = w;
        }

        /// <summary>
        /// 
        /// </summary>
        public int N { get; }

        /// <summary>
        /// 
        /// </summary>
        public int C { get; }

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
        /// <param name="n"></param>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        public T At(int n, int c, int h, int w)
        {
            //w=n, h=c, d=h, n=w

            //var index = ((n * Depth + d) * Width + h) * Height + w;
            var index = ((w * H + h) * N + c) * C + n;
            return m_tensor.Data[index];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="n"></param>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="value"></param>
        public void At(int n, int c, int h, int w, T value)
        {
            //var index = x + y * DimXCount + z * DimXCount * DimYCount + n * DimXCount * DimYCount * DimZCount;

            // width offset = x
            // height offset: y * DimXCount
            // chan offset:  z * DimXCount * DimYCount
            // batch off set: n * DimXCount * DimYCount * DimZCount;

            var index = ((w * H + h) * N + c) * C + n;
            m_tensor.Data[index] = value;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeN(int c, int h, int w, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(i, c, h, w);
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="n"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeC(int n, int h, int w, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(n, i, h, w);
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="n"></param>
        /// <param name="c"></param>
        /// <param name="w"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeH(int n, int c, int w, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(n, c, i, w);
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="n"></param>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        public void RangeW(int n, int c, int h, Interval1D interval, T[] output)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                output[i] = At(n, c, h, i);
            }
        }
    }
}
