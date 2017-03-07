using System;
using System.Linq;
using SharpLearning.Containers.Views;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class Tensor<T> : IEquatable<Tensor<T>>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="layout"></param>
        public Tensor(TensorShape shape, DataLayout layout)
            : this(new T[shape.ElementCount], shape, layout)
        {
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
        /// <param name="data"></param>
        /// <param name="shape"></param>
        /// <param name="layout"></param>
        public Tensor(T[] data, TensorShape shape, DataLayout layout)
        {
            if (data == null) { throw new ArgumentNullException(nameof(data)); }
            if (data.Length != shape.ElementCount)
            { throw new ArgumentNullException($"data length: {data.Length} does not match shape size: {shape.ElementCount}"); }
            
            Shape = shape;
            Layout = layout;
            Data = data;

            DimensionOffSets = new int[Dimensions.Length - 1];
            for (int i = 0; i < Dimensions.Length - 1; i++)
            {
                DimensionOffSets[i] = Dimensions.Skip(i + 1).Aggregate((x, y) => x * y);
            }
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
        public int[] DimensionOffSets { get;  }

        /// <summary>
        /// 
        /// </summary>
        public int DimensionCount { get { return Shape.Dimensions.Length; } }

        /// <summary>
        /// 
        /// </summary>
        public int ElementCount { get { return Shape.ElementCount; } }

        /// <summary>
        /// Returns a slice of the tensor specified by the interval.
        /// values are copied.
        /// </summary>
        /// <param name="interval"></param>
        /// <returns></returns>
        public Tensor<T> Slice(Interval1D interval)
        {
            // only works for first dimension currently.
            int dimension = 0;

            if (dimension >= Dimensions.Length)
            {
                throw new ArgumentException($"Dimension: {dimension} is larger than dimension count: {DimensionCount}");
            }
                        
            var sliceShape = new TensorShape(Dimensions.Skip(dimension).ToArray());
            var sliceMaxElementCount = Dimensions.Skip(dimension).Aggregate((x, y) => x * y);

            if (sliceShape.ElementCount > sliceMaxElementCount)
            {
                throw new ArgumentException("Slice elementCount larger than availible elemens in tensor for this dimension");
            }

            var sliceData = new T[sliceShape.ElementCount];
            var startIndex = interval.FromInclusive * DimensionOffSets[0];
            var length = interval.Length * DimensionOffSets[0];

            Array.Copy(Data, startIndex, sliceData, 0, length);

            return new Tensor<T>(sliceData, sliceShape, DataLayout.RowMajor);
        }

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
        /// <param name="data"></param>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Tensor<T> CreateRowMajor(T[] data, params int[] dimensions)
        {
            return new Tensor<T>(data, new TensorShape(dimensions), DataLayout.RowMajor);
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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(Tensor<T> other)
        {
            if (this.Shape != other.Shape) { return false; }
            if (!this.Data.SequenceEqual(other.Data)) { return false; }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            Tensor<T> other = obj as Tensor<T>;
            if (other != null && Equals(other))
            {
                return true;
            }

            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                int hash = 17;
                hash = hash * 23 + Data.GetHashCode();
                hash = hash * 23 + Shape.GetHashCode();

                return hash;
            }
        }
    }
}
