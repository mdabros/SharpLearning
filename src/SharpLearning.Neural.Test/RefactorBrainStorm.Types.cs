using System;
using System.Linq;
using SharpLearning.Containers.Views;

namespace SharpLearning.Neural.Test.RefactorBranStorm
{
    /// <summary>
    /// 
    /// </summary>
    public enum DataLayout
    {
        /// <summary>
        /// 
        /// </summary>
        RowMajor,

        /// <summary>
        /// 
        /// </summary>
        ColumnMajor
    }

    /// <summary>
    /// 
    /// </summary>
    public struct TensorShape : IEquatable<TensorShape>
    {
        /// <summary>
        /// 
        /// </summary>
        public TensorShape(params int[] dimensions)
        {
            if (dimensions == null) { throw new ArgumentNullException(); }
            Dimensions = dimensions;

            DimensionOffSets = new int[Dimensions.Length - 1];
            for (int i = 0; i < Dimensions.Length - 1; i++)
            {
                DimensionOffSets[i] = Dimensions.Skip(i + 1).Aggregate((x, y) => x * y);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int[] Dimensions { get; }

        /// <summary>
        /// 
        /// </summary>
        public int[] DimensionOffSets { get; }

        /// <summary>
        /// 
        /// </summary>
        public int ElementCount { get { return Elements(); } }

        /// <summary>
        /// 
        /// </summary>
        public int Rank { get { return Dimensions.Length; } }


        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        int Elements()
        {
            if (Dimensions.Length == 0)
                return 0;

            if (Dimensions.Sum() == 0)
                return 0;

            var elements = 1;
            foreach (var dimension in Dimensions)
            {
                elements *= dimension;
            }

            return elements;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(TensorShape other)
        {
            if (Dimensions.Length != other.Dimensions.Length)
            {
                return false;
            }

            for (int i = 0; i < Dimensions.Length; i++)
            {
                if (Dimensions[i] != other.Dimensions[i])
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator ==(TensorShape p1, TensorShape p2)
        {
            return p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator !=(TensorShape p1, TensorShape p2)
        {
            return !p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (obj is TensorShape)
                return Equals((TensorShape)obj);
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
                hash = hash * 23 + Dimensions.GetHashCode();

                return hash;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            var dimensions = "TensorShape(";
            for (int i = 0; i < Dimensions.Length; i++)
            {
                dimensions += $"{Dimensions[i]}, ";
            }
            dimensions += ")";

            return dimensions;
        }
    }

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
            { throw new ArgumentException($"data length: {data.Length} does not match shape size: {shape.ElementCount}"); }

            Shape = shape;
            Layout = layout;
            Data = data;
        }

        /// <summary>
        /// 
        /// </summary>
        public T[] Data { get; }

        /// <summary>
        /// 
        /// </summary>
        public TensorShape Shape { get; private set; }

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
        public int[] DimensionOffSets { get { return Shape.DimensionOffSets; } }

        /// <summary>
        /// 
        /// </summary>
        public int Rank { get { return Shape.Rank; } }

        /// <summary>
        /// 
        /// </summary>
        public int ElementCount { get { return Shape.ElementCount; } }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        public void Reshape(TensorShape shape)
        {
            if (Shape.ElementCount != shape.ElementCount)
            {
                throw new ArgumentException($"Current element count {shape.ElementCount} differs from new element count {shape.ElementCount}");
            }

            Shape = shape;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="interval"></param>
        /// <returns></returns>
        public Tensor<T> SliceCopy(Interval1D interval)
        {
            return SliceCopy(interval.FromInclusive, interval.Length);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="fromInclusive"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        public Tensor<T> SliceCopy(int fromInclusive, int length = 1)
        {
            // only works for first dimension currently                       
            var sliceShape = new TensorShape(Dimensions.Skip(1).ToArray());
            var sliceMaxElementCount = Dimensions.Skip(1).Aggregate((x, y) => x * y);

            if (sliceShape.ElementCount > sliceMaxElementCount)
            {
                throw new ArgumentException("Slice elementCount larger than available elements in tensor for this dimension");
            }

            var slice = new Tensor<T>(sliceShape, DataLayout.RowMajor);
            SliceCopy(fromInclusive, length, slice);

            return slice;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="fromInclusive"></param>
        /// <param name="length"></param>
        /// <param name="copy"></param>
        /// <returns></returns>
        public void SliceCopy(int fromInclusive, int length, Tensor<T> copy)
        {
            // only works for first dimension currently.
            int dimension = 0;

            if (dimension >= Dimensions.Length)
            {
                throw new ArgumentException($"Dimension: {dimension} is larger than dimension count: {Rank}");
            }

            int elementsToCopy = length * DimensionOffSets[0];
            if (elementsToCopy != copy.ElementCount)
            {
                throw new ArgumentException($"Required element count : {elementsToCopy} is larger than copy element count: {copy.ElementCount}");
            }

            var dimensionSize = DimensionOffSets[dimension];

            var index = fromInclusive;
            var srcStartIndex = index * dimensionSize;

            Array.Copy(Data, srcStartIndex, copy.Data, 0, elementsToCopy);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="copy"></param>
        public void SliceCopy(int[] indices, Tensor<T> copy)
        {
            // only works for first dimension currently.
            int dimension = 0;
            int elements = indices.Length * DimensionOffSets[0];

            if (elements != copy.ElementCount)
            {
                throw new ArgumentException($"Required element count : {elements} is larger than copy element count: {copy.ElementCount}");
            }

            var dimensionSize = DimensionOffSets[dimension];
            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                var startIndex = index * dimensionSize;
                var copyStartIndex = i * dimensionSize;

                Array.Copy(Data, startIndex, copy.Data, copyStartIndex, dimensionSize);
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <param name="values"></param>
        public void SetSlice(int index, Tensor<T> values)
        {
            // only works for first dimension currently.
            int dimension = 0;
            int length = values.Dimensions[0];

            var dstDimensionSize = DimensionOffSets[dimension];
            var srcDimensionSize = values.DimensionOffSets[dimension];

            int elementsToCopy = length * values.DimensionOffSets[0];
            if (dstDimensionSize != srcDimensionSize)
            {
                throw new ArgumentException($"Destination dimension size: {dstDimensionSize} differs from source dimension size {srcDimensionSize}");
            }

            var dstStartIndex = index * dstDimensionSize;

            Array.Copy(values.Data, 0, Data, dstStartIndex, elementsToCopy);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Tensor<T> Build(T value, params int[] dimensions)
        {
            var tensor = new Tensor<T>(dimensions, DataLayout.RowMajor);
            tensor.Map(v => value);
            return tensor;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Tensor<T> Build(params int[] dimensions)
        {
            return new Tensor<T>(dimensions, DataLayout.RowMajor);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="data"></param>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Tensor<T> Build(T[] data, params int[] dimensions)
        {
            return new Tensor<T>(data, new TensorShape(dimensions), DataLayout.RowMajor);
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
