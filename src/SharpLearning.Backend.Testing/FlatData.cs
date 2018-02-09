using System;

namespace SharpLearning.Backend.Testing
{
    public readonly struct FlatData<T>
    {
        public FlatData(int[] shape, T[] data)
        {
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Data = data ?? throw new ArgumentNullException(nameof(data));
            var sum = Shape.Product();
            if (sum != Data.Length)
            {
                throw new ArgumentException($"Shape sum {sum} not equal")
            }
        }

        public int[] Shape { get; }
        public T[] Data { get; }
    }
}
