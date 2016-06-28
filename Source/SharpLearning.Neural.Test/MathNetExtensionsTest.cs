using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MathNet.Numerics.LinearAlgebra;
using System.Diagnostics;

namespace SharpLearning.Neural.Test
{
    [TestClass]
    public class MathNetExtensionsTest
    {
        [TestMethod]
        public void MathNetExtensions_Add()
        {
            var matrix = Matrix<float>.Build.Dense(2, 3);
            var vector = Vector<float>.Build.Dense(new float[] { 1f, 2f, 3f });
            var actual = Matrix<float>.Build.Dense(2, 3);

            matrix.Add(vector, actual);

            Trace.WriteLine(string.Join(", ", actual.ToColumnWiseArray()));
            Trace.WriteLine(actual.ToString());

            var expected = Matrix<float>.Build.Dense(2, 3, new float[] { 1, 1, 2, 2, 3, 3 });
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }

        [TestMethod]
        public void MathNetExtensions_Multiply()
        {
            var matrix = Matrix<float>.Build.Dense(2, 3, 1);
            var vector = Vector<float>.Build.Dense(new float[] { 1f, 0f, 1f });
            var actual = Matrix<float>.Build.Dense(2, 3);

            matrix.Multiply(vector, actual);

            Trace.WriteLine(string.Join(", ", actual.ToColumnWiseArray()));
            Trace.WriteLine(actual.ToString());

            var expected = Matrix<float>.Build.Dense(2, 3, new float[] { 1, 1, 0, 0, 1, 1 });
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }

        [TestMethod]
        public void MathNetExtensions_ColumnWiseMean()
        {
            var matrix = Matrix<float>.Build.Dense(3, 3, new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var actual = Vector<float>.Build.Dense(3);
            
            matrix.ColumnWiseMean(actual);

            Trace.WriteLine(string.Join(", ", actual));
            Trace.WriteLine(matrix.ToString());

            var expected = Vector<float>.Build.Dense(new float[] { 2, 5, 8 });
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }

        [TestMethod]
        public void MathNetExtensions_ColumnWiseMean_2()
        {
            var matrix = Matrix<float>.Build.Dense(4, 2, new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            var actual = Vector<float>.Build.Dense(3);

            matrix.ColumnWiseMean(actual);

            Trace.WriteLine(string.Join(", ", actual));
            Trace.WriteLine(matrix.ToString());

            var expected = Vector<float>.Build.Dense(new float[] { 2.5f, 6.5f, 0f });
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
    }
}
