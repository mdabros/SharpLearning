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
        /// <param name="w"></param>
        public TensorIndexer1D(Tensor<T> tensor, int w)
        {
            if (tensor == null)
            { throw new ArgumentNullException(nameof(tensor)); }

            Shape = new TensorShape(w);
            if (Shape != tensor.Shape)
            {
                throw new ArgumentException($"Indexer shape: {Shape} does not match tensor shape: {tensor.Shape}");
            }

            m_tensor = tensor;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <returns></returns>
        public T At(int w)
        {
            return m_tensor.Data[w];
        }

        /// <summary>
        /// 
        /// </summary>
        public TensorShape Shape { get; }

        /// <summary>
        /// 
        /// </summary>
        public int NumberOfElements { get { return Shape.ElementCount; } }
    }
}
