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
        /// <param name="size"></param>
        /// <returns></returns>
        public ITensorIndexer1D<T> AsTensor1D(int size)
        {
            return new TensorIndexer1D<T>(this, size);
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
        public ITensorIndexer2D<T> AsTensor2D(int width, int height)
        {
            switch (Layout)
            {
                case DataLayout.RowMajor:
                    return new RowMajorTensorIndexer2D<T>(this, width, height);
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
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="depth"></param>
        /// <returns></returns>
        public ITensorIndexer3D<T> AsTensor3D(int width, int height, int depth)
        {
            switch (Layout)
            {
                case DataLayout.RowMajor:
                    return new RowMajorTensorIndexer3D<T>(this, width, height, depth);
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
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="depth"></param>
        /// <param name="dim4"></param>
        /// <returns></returns>
        public ITensorIndexer4D<T> AsTensor4D(int width, int height, int depth, int dim4)
        {
            switch (Layout)
            {
                case DataLayout.RowMajor:
                    return new RowMajorTensorIndexer4D<T>(this, width, height, depth, dim4);
                case DataLayout.ColumnMajor:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException("Unknown DataLayout: " + Layout);
            }
        }
    }
}
