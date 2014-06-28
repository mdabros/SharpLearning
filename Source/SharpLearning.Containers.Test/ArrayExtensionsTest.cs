using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using System;
using System.Linq;

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

        [TestMethod]
        public void ArrayExtensions_SortWith()
        {
            var values = new int[] { 0, 1, 2, 3, 4, 5 };
            var keys  = new int[] { 5, 4, 3, 2, 1, 0 };
            var interval = Interval1D.Create(0, keys.Length);

            keys.SortWith(interval, values);
            var expectedKeys = new int[] { 0, 1, 2, 3, 4, 5 };
            CollectionAssert.AreEqual(expectedKeys, keys);

            var expectedValues = new int[] { 5, 4, 3, 2, 1, 0 };
            CollectionAssert.AreEqual(expectedValues, values);
        }

        [TestMethod]
        public void ArrayExtensions_SortWith_Interval()
        {
            var values = new int[] { 0, 1, 2, 3, 4, 5 };
            var keys = new int[] { 5, 4, 3, 2, 1, 0 };
            var interval = Interval1D.Create(2, keys.Length);

            keys.SortWith(interval, values);
            var expectedKeys = new int[] { 5, 4, 0, 1, 2, 3 };
            CollectionAssert.AreEqual(expectedKeys, keys);

            var expectedValues = new int[] { 0, 1, 5, 4, 3, 2 };
            CollectionAssert.AreEqual(expectedValues, values);
        }

        [TestMethod]
        public void ArrayExtensions_CopyTo()
        {
            var values = new int[] { 0, 1, 2, 3, 4, 5 };
            var interval = Interval1D.Create(0, values.Length);

            var destination = new int[interval.Length];

            values.CopyTo(interval, destination);
            CollectionAssert.AreEqual(destination, values);
        }

        [TestMethod]
        public void ArrayExtensions_CopyTo_Interval()
        {
            var values = new int[] { 0, 1, 2, 3, 4, 5 };
            var interval = Interval1D.Create(1, 5);

            var destination = new int[values.Length];

            values.CopyTo(interval, destination);
            var expected = new int[] {0, 1, 2, 3, 4, 0 };
            CollectionAssert.AreEqual(expected, destination);
        }

        [TestMethod]
        public void ArrayExtensions_IndexedCopy()
        {
            var values = new int[] { 0, 10, 20, 30, 40, 50 };
            var indices = new int[] { 1, 1, 2, 2, 2, 5 };
            var interval = Interval1D.Create(0, values.Length);

            var destination = new int[values.Length];

            indices.IndexedCopy(values, interval, destination);
            var expected = new int[] { 10, 10, 20, 20, 20, 50 };
            CollectionAssert.AreEqual(expected, destination);
        }

        [TestMethod]
        public void ArrayExtensions_IndexedCopy_Interval()
        {
            var values = new int[] { 0, 10, 20, 30, 40, 50 };
            var indices = new int[] { 1, 1, 2, 2, 2, 5 };
            var interval = Interval1D.Create(1, 5);

            var destination = new int[values.Length];

            indices.IndexedCopy(values, interval, destination);
            var expected = new int[] { 0, 10, 20, 20, 20, 0 };
            CollectionAssert.AreEqual(expected, destination);
        }

        [TestMethod]
        public void ArrayExtensions_IndexedCopy_ColumnView()
        {
            var values = new double[] { 0, 10, 20, 30, 40, 50 };
            var matrix = new F64Matrix(values, 6, 1);
            var indices = new int[] { 1, 1, 2, 2, 2, 5 };
            var destination = new double[values.Length];
            var interval = Interval1D.Create(0, values.Length);

            using (var ptr = matrix.GetPinnedPointer())
            {
                var view = ptr.View().ColumnView(0);
                indices.IndexedCopy(view, interval, destination);
                var expected = new double[] { 10, 10, 20, 20, 20, 50 };
                CollectionAssert.AreEqual(expected, destination);
            }
        }

        [TestMethod]
        public void ArrayExtensions_IndexedCopy_ColumnView_Interval()
        {
            var values = new double[] { 0, 10, 20, 30, 40, 50 };
            var matrix = new F64Matrix(values, 6, 1);
            var indices = new int[] { 1, 1, 2, 2, 2, 5 };
            var destination = new double[values.Length];
            var interval = Interval1D.Create(1, 5);

            using (var ptr = matrix.GetPinnedPointer())
            {
                var view = ptr.View().ColumnView(0);
                indices.IndexedCopy(view, interval, destination);
                var expected = new double[] { 0, 10, 20, 20, 20, 0 };
                CollectionAssert.AreEqual(expected, destination);
            }
        }

        [TestMethod]
        public void ArrayExtensions_Shuffle()
        {
            var actual = Enumerable.Range(0, 10).ToArray();
            actual.Shuffle(new Random(42));

            var expected = new int[] { 9, 0, 4, 2, 5, 7, 3, 8, 1, 6 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ArrayExtensions_Stratify_Even()
        {
            var actual = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            actual.Stratify(actual, new Random(42), 2);

            var expected = new int[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ArrayExtensions_Stratify_Uneven()
        {
            var actual = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            actual.Stratify(actual, new Random(42), 3);

            var expected = new int[] { 1, 1, 1, 2, 2, 3, 1, 1, 1, 2, 2, 3, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ArrayExtensions_Stratify_Too_Many_Folds()
        {
            var actual = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            actual.Stratify(actual, new Random(42), 10);

            var expected = new int[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, };
            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
