using System;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class TensorIndexer1D<T> : ITensorIndexer1D<T>
    {
        Tensor<T> m_tensor;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="length"></param>
        public TensorIndexer1D(Tensor<T> tensor, int length)
        {
            if (tensor == null)
            { throw new ArgumentNullException(nameof(tensor)); }

            Shape = new TensorShape(length);
            if (Shape != tensor.Shape)
            {
                throw new ArgumentException($"Indexer shape: {Shape} does not match tensor shape: {tensor.Shape}");
            }

            m_tensor = tensor;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public T At(int index)
        {
            return m_tensor.Data[index];
        }

        /// <summary>
        /// 
        /// </summary>
        public TensorShape Shape { get; }

        /// <summary>
        /// 
        /// </summary>
        public int NumberOfElements { get { return Shape.NumberOfElements; } }
    }
}
