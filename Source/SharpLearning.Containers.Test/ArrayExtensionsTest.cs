using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Containers.Test
{
    [TestClass]
    public class ArrayExtensionsTest
    {
        [TestMethod]
        public void ArrayExtensions_GetIndices()
        {
            var sut = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var actual = sut.GetIndices(new int[] { 0, 4, 8 });
            var expected = new double[] { 1, 5, 9 };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ArrayExtensions_AsString()
        {
            var sut = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var actual = sut.AsString();
            var expected = new string[] { "1", "2", "3", "4", "5", "6", "7", "8", "9" };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ArrayExtensions_AsF64()
        {
            var sut = new string[] { "1", "2", "3", "4", "5", "6", "7", "8", "9" }; 
            var actual = sut.AsF64();
            var expected = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ArrayExtensions_AsInt32()
        {
            var sut = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var actual = sut.AsInt32();
            var expected = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
