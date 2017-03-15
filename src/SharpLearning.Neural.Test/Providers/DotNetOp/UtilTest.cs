using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Arithmetic;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.Providers.DotNetOp;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace SharpLearning.Neural.Test.Providers.DotNetOp
{
    [TestClass]
    public class UtilTest
    {
        [TestMethod]
        public void Util_Multiply()
        {
            var a = Tensor<float>.Build(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
            var b = Tensor<float>.Build(new float[] { 7, 8, 9, 10, 11, 12 }, 3, 2);

            var actual = Tensor<float>.Build(a.Dimensions[0], b.Dimensions[1]);

            Utils.Multiply(a, b, actual);

            var expected = Tensor<float>.Build(new float[] { 58, 64, 139, 154 }, 2, 2);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void Util_Multiply_Timing()
        {
            var elements = 300;
            var iterations = 10;
            var timer = new Stopwatch();

            Tensor(elements, iterations, timer);

            Matrix(elements, iterations, timer);

            MathNet_MKL(elements, iterations, timer);

            Tensor_Mathnet_MKL(elements, iterations, timer);
        }

        private static void MathNet_MKL(int elements, int iterations, Stopwatch timer)
        {
            timer.Reset();
            var m1 = Matrix<float>.Build.Dense(elements, elements);
            var m2 = Matrix<float>.Build.Dense(elements, elements);
            var mOut = Matrix<float>.Build.Dense(elements, elements);

            m1.Multiply(m2, mOut);

            for (int i = 0; i < iterations; i++)
            {
                timer.Start();
                m1.Multiply(m2, mOut);
                timer.Stop();
            }

            Trace.WriteLine($"Math.net MKL: " + timer.ElapsedMilliseconds);
        }


        private static void Matrix(int elements, int iterations, Stopwatch timer)
        {
            timer.Reset();
            var m1 = new F64Matrix(elements, elements);
            var m2 = new F64Matrix(elements, elements);
            var mOut = new F64Matrix(elements, elements);

            MatrixMultiplication.MultiplyF64(m1, m2, mOut);

            for (int i = 0; i < iterations; i++)
            {
                timer.Start();
                MatrixMultiplication.MultiplyF64(m1, m2, mOut);
                timer.Stop();
            }

            Trace.WriteLine($"Matrix: " + timer.ElapsedMilliseconds);
        }

        private static void Tensor(int elements, int iterations, Stopwatch timer)
        {
            timer.Reset();
            var t1 = Tensor<float>.Build(elements, elements);
            var t2 = Tensor<float>.Build(elements, elements);
            var tOut = Tensor<float>.Build(elements, elements);

            Utils.Multiply(t1, t2, tOut);

            for (int i = 0; i < iterations; i++)
            {
                timer.Start();
                Utils.Multiply(t1, t2, tOut);
                timer.Stop();
            }

            Trace.WriteLine($"Tensor: " + timer.ElapsedMilliseconds);
        }

        private static void Tensor_Mathnet_MKL(int elements, int iterations, Stopwatch timer)
        {
            timer.Reset();
            var t1 = Tensor<float>.Build(elements, elements);
            var t2 = Tensor<float>.Build(elements, elements);
            var tOut = Tensor<float>.Build(elements, elements);

            Utils.Multiply_MathNet(t1, t2, tOut);

            for (int i = 0; i < iterations; i++)
            {
                timer.Start();
                Utils.Multiply_MathNet(t1, t2, tOut);
                timer.Stop();
            }

            Trace.WriteLine($"Tensor Math.et MKL: " + timer.ElapsedMilliseconds);
        }
    }
}
