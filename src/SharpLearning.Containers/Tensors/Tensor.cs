using System;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class Tensor<T>
    {
        T[] m_storage;

        /// <summary>
        /// 
        /// </summary>
        public TensorShape Shape { get; }

        /// <summary>
        /// 
        /// </summary>
        public DataLayout Layout { get; }

        /// <summary>
        /// 
        /// </summary>
        public Type DataType { get { return typeof(T); } }

        /// <summary>
        /// 
        /// </summary>
        public int[] Dimensions { get { return Shape.Dimensions; } }

        /// <summary>
        /// 
        /// </summary>
        public int NumberOfElements { get { return Shape.NumberOfElements; } }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="layout"></param>
        public Tensor(TensorShape shape, DataLayout layout)
        {
            Shape = shape;
            Layout = layout;
            m_storage = new T[shape.NumberOfElements];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="layout"></param>
        public Tensor(int[] dimensions, DataLayout layout)
            : this(new TensorShape(dimensions), layout)
        {
        }
    }
}
