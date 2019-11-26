using System;

namespace SharpLearning.Neural.Test.Backends
{
    public abstract class Tensor
    {
        public readonly IBackend Backend;

        public abstract string name { get; }

        public DataType? dtype
        {
            get { return Backend.dtype(this); }
        }

        public Tensor(IBackend backend)
        {
            Backend = backend ?? throw new ArgumentNullException(nameof(backend));
        }

        public int?[] Shape
        {
            get { return Backend.Shape(this); }
        }

        public Tensor Transpose()
        {
            return Backend.Transpose(this);
        }

        public static Tensor operator *(double a, Tensor b)
        {
            return b.Backend.Multiply(a, b);
        }

        public static Tensor operator *(Tensor a, int b)
        {
            return a.Backend.Multiply(a, b);
        }

        public static Tensor operator *(int a, Tensor b)
        {
            return b.Backend.Multiply(a, b);
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            return b.Backend.Multiply(a, b);
        }

        public static Tensor operator /(double a, Tensor b)
        {
            return b.Backend.Divide(a, b);
        }

        public static Tensor operator /(Tensor a, int b)
        {
            return a.Backend.Divide(a, b);
        }

        public static Tensor operator /(int a, Tensor b)
        {
            return b.Backend.Divide(a, b);
        }

        public static Tensor operator /(Tensor a, Tensor b)
        {
            return b.Backend.Divide(a, b);
        }

        public static Tensor operator +(double a, Tensor b)
        {
            return b.Backend.Add(a, b);
        }

        public static Tensor operator +(int a, Tensor b)
        {
            return b.Backend.Add(a, b);
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            return b.Backend.Add(a, b);
        }

        public static Tensor operator +(Tensor a, double b)
        {
            return a.Backend.Add(a, b);
        }

        public static Tensor operator -(double a, Tensor b)
        {
            return b.Backend.Subtract(a, b);
        }

        public static Tensor operator -(Tensor a, Tensor b)
        {
            return b.Backend.Subtract(a, b);
        }
    }
}
