using System;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class Tensor<T>
    {
        ITensorIndexer1D<T> m_indexer1D;
        ITensorIndexer2D<T> m_indexer2D;
        ITensorIndexer3D<T> m_indexer3D;
        ITensorIndexer4D<T> m_indexer4D;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="layout"></param>
        public Tensor(TensorShape shape, DataLayout layout)
        {
            Shape = shape;
            Layout = layout;
            Data = new T[shape.NumberOfElements];
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

        /// <summary>
        /// 
        /// </summary>
        public T[] Data { get; }
        
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
        public int NumberOfDimensions { get { return Shape.Dimensions.Length; } }

        /// <summary>
        /// 
        /// </summary>
        public int NumberOfElements { get { return Shape.NumberOfElements; } }

        /// <summary>
        /// 
        /// </summary>
        public ITensorIndexer1D<T> Indexer1D
        {
            get
            {
                if (m_indexer1D != null)
                {
                    return m_indexer1D;
                }
                else
                {
                    m_indexer1D = Create1DIndexer();
                    return m_indexer1D;
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public ITensorIndexer2D<T> Indexer2D
        {
            get
            {
                if(m_indexer2D != null)
                {
                    return m_indexer2D;
                }
                else
                {
                    m_indexer2D = Create2DIndexer();
                    return m_indexer2D;
                }
            }
        } 

        /// <summary>
        /// 
        /// </summary>
        public ITensorIndexer3D<T> Indexer3D
        {
            get
            {
                if (m_indexer3D != null)
                {
                    return m_indexer3D;
                }
                else
                {
                    m_indexer3D = Create3DIndexer();
                    return m_indexer3D;
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public ITensorIndexer4D<T> Indexer4D
        {
            get
            {
                if (m_indexer4D != null)
                {
                    return m_indexer4D;
                }
                else
                {
                    m_indexer4D = Create4DIndexer();
                    return m_indexer4D;
                }
            }
        }

        ITensorIndexer1D<T> Create1DIndexer()
        {
            return new TensorIndexer1D<T>(this, Dimensions[0]);
        }

        ITensorIndexer2D<T> Create2DIndexer()
        {
            switch (Layout)
            {
                case DataLayout.RowMajor:
                    return new RowMajorTensorIndexer2D<T>(this, Dimensions[0], Dimensions[1]);
                case DataLayout.ColumnMajor:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException("Unknown DataLayout: " + Layout);
            }
        }

        ITensorIndexer3D<T> Create3DIndexer()
        {
            switch (Layout)
            {
                case DataLayout.RowMajor:
                    return new RowMajorTensorIndexer3D<T>(this, Dimensions[0], Dimensions[1], Dimensions[2]);
                case DataLayout.ColumnMajor:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException("Unknown DataLayout: " + Layout);
            }
        }

        ITensorIndexer4D<T> Create4DIndexer()
        {
            switch (Layout)
            {
                case DataLayout.RowMajor:
                    return new RowMajorTensorIndexer4D<T>(this, Dimensions[0], Dimensions[1], Dimensions[2], Dimensions[3]);
                case DataLayout.ColumnMajor:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException("Unknown DataLayout: " + Layout);
            }
        }
    }
}
