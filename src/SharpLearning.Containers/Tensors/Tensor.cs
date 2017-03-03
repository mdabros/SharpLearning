using System;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class Tensor<T>
    {
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
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Tensor<T> CreateRowMajor(params int[] dimensions)
        {
            return new Tensor<T>(dimensions, DataLayout.RowMajor);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name=""></param>
        /// <returns></returns>
        public ITensorIndexer1D<T> AsTensor1D()
        {
            return AsTensor1D(Dimensions[0]);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dim"></param>
        /// <returns></returns>
        public ITensorIndexer1D<T> AsTensor1D(int dim)
        {
            return new TensorIndexer1D<T>(this, dim);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public ITensorIndexer2D<T> AsTensor2D()
        {
            return AsTensor2D(Dimensions[0], Dimensions[1]);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public ITensorIndexer2D<T> AsTensor2D(int dimX, int dimY)
        {
            switch (Layout)
            {
                case DataLayout.RowMajor:
                    return new RowMajorTensorIndexer2D<T>(this, dimX, dimY);
                case DataLayout.ColumnMajor:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException("Unknown DataLayout: " + Layout);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public ITensorIndexer3D<T> AsTensor3D()
        {
            return AsTensor3D(Dimensions[0], Dimensions[1], Dimensions[2]);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimX"></param>
        /// <param name="dimY"></param>
        /// <param name="dimZ"></param>
        /// <returns></returns>
        public ITensorIndexer3D<T> AsTensor3D(int dimX, int dimY, int dimZ)
        {
            switch (Layout)
            {
                case DataLayout.RowMajor:
                    return new RowMajorTensorIndexer3D<T>(this, dimX, dimY, dimZ);
                case DataLayout.ColumnMajor:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException("Unknown DataLayout: " + Layout);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public ITensorIndexer4D<T> AsTensor4D()
        {
            return AsTensor4D(Dimensions[0], Dimensions[1], Dimensions[2], Dimensions[3]);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimX"></param>
        /// <param name="dimY"></param>
        /// <param name="dimZ"></param>
        /// <param name="dimN"></param>
        /// <returns></returns>
        public ITensorIndexer4D<T> AsTensor4D(int dimX, int dimY, int dimZ, int dimN)
        {
            switch (Layout)
            {
                case DataLayout.RowMajor:
                    return new RowMajorTensorIndexer4D<T>(this, dimX, dimY, dimZ, dimN);
                case DataLayout.ColumnMajor:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException("Unknown DataLayout: " + Layout);
            }
        }
    }
}
