using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Arithmetic;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.Providers.DotNetOp;
using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural.Test.Providers.DotNetOp
{
    [TestClass]
    public class UtilTest
    {
        [TestMethod]
        public void Util_Multiply()
        {
            var a = Tensor<float>.CreateRowMajor(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3)
                .AsTensor2D();
            var b = Tensor<float>.CreateRowMajor(new float[] { 7, 8, 9, 10, 11, 12 }, 3, 2)
                .AsTensor2D();

            var actual = Tensor<float>.CreateRowMajor(a.H, b.W)
                .AsTensor2D();

            Utils.Multiply(a, b, actual);

            var expected = Tensor<float>.CreateRowMajor(new float[] { 58, 64, 139, 154 }, 2, 2)
                .AsTensor2D();

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
        }

        private static void MathNet_MKL(int elements, int iterations, Stopwatch timer)
        {
            timer.Reset();
            var m1 = Matrix<float>.Build.Dense(elements, elements);
            var m2 = Matrix<float>.Build.Dense(elements, elements);
            var mOut = Matrix<float>.Build.Dense(elements, elements);

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
            var t1 = Tensor<float>.CreateRowMajor(elements, elements)
                .AsTensor2D();
            var t2 = Tensor<float>.CreateRowMajor(elements, elements)
                .AsTensor2D();
            var tOut = Tensor<float>.CreateRowMajor(elements, elements)
                .AsTensor2D();

            for (int i = 0; i < iterations; i++)
            {
                timer.Start();
                Utils.Multiply(t1, t2, tOut);
                timer.Stop();
            }

            Trace.WriteLine($"Tensor: " + timer.ElapsedMilliseconds);
        }
    }
}
