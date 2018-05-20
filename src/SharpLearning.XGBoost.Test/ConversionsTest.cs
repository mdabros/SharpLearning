using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.XGBoost.Test
{
    [TestClass]
    public class ConversionsTest
    {
        [TestMethod]
        public void Conversions_ToFloatJaggedArray()
        {
            var data = new double[]
            {
                10, 11,
                12, 13,
                14, 15
            };
            var matrix = new F64Matrix(data, 3, 2);
            var actual = matrix.ToFloatJaggedArray();
            var expected = new float[][]
            {
                new float[] { 10, 11 },
                new float[] { 12, 13 },
                new float[] { 14, 15 },
            };

            AssertArrays(actual, expected);
        }

        [TestMethod]
        public void Conversions_ToFloatJaggedArray_Indexed_All()
        {
            var data = new double[]
            {
                10, 11,
                12, 13,
                14, 15
            };
            var matrix = new F64Matrix(data, 3, 2);
            var actual = matrix.ToFloatJaggedArray(new int[] { 0, 1, 2 });
            var expected = new float[][]
            {
                new float[] { 10, 11 },
                new float[] { 12, 13 },
                new float[] { 14, 15 },
            };

            AssertArrays(actual, expected);
        }

        [TestMethod]
        public void Conversions_ToFloatJaggedArray_Indexed()
        {
            var data = new double[]
            {
                10, 11,
                12, 13,
                14, 15
            };
            var matrix = new F64Matrix(data, 3, 2);
            var actual = matrix.ToFloatJaggedArray(new int[] { 0, 2 });
            var expected = new float[][]
            {
                new float[] { 10, 11 },
                new float[] { 14, 15 },
            };

            AssertArrays(actual, expected);
        }

        [TestMethod]
        public void ToFloat()
        {
            var data = new double[] { 5, 4, 2, 1 };
            var actual = data.ToFloat();
            var expected = new float[] { 5, 4, 2, 1 };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ToFloat_indexed()
        {
            var data = new double[] { 5, 4, 2, 1 };
            var actual = data.ToFloat(new int[] { 0, 3 });
            var expected = new float[] { 5, 1 };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ToDouble()
        {
            var data = new float[] { 5, 4, 2, 1 };
            var actual = data.ToDouble();
            var expected = new double[] { 5, 4, 2, 1 };

            CollectionAssert.AreEqual(expected, actual);
        }

        static void AssertArrays(float[][] actual, float[][] expected)
        {
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                CollectionAssert.AreEqual(expected[i], actual[i]);
            }
        }
    }
}
