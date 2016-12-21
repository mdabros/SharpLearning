using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Neural.Test
{
    public static class MatrixAsserts
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="expected"></param>
        /// <param name="actual"></param>
        /// <param name="delta"></param>
        public static void AreEqual(Matrix<float> expected, Matrix<float> actual, double delta = 0.0001)
        {
            var m1Array = expected.ToRowWiseArray();
            var m2Array = actual.ToRowWiseArray();

            Assert.AreEqual(m1Array.Length, m2Array.Length);

            for (int i = 0; i < m1Array.Length; i++)
            {
                Assert.AreEqual(m1Array[i], m2Array[i], delta);
            }
        }

        public static void AreEqual(Vector<float> expected, Vector<float> actual, double delta = 0.0001)
        {
            AreEqual(expected.Data(), actual.Data(), delta);
        }

        public static void AreEqual(float[] expected, float[] actual, double delta = 0.0001)
        {
            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], delta);
            }
        }
    }
}
