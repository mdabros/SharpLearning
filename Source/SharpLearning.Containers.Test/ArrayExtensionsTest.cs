using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using System;
using System.Collections.Generic;
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
        public void ArrayExtensions_Sum()
        {
            var values = new double[] { 0, 10, 20, 30, 40, 50 };
            var actual = values.Sum(Interval1D.Create(1, 5));

            Assert.AreEqual(100, actual);
        }

        [TestMethod]
        public void ArrayExtensions_Exp()
        {
            var values = new List<double[]> {
                new double[] { 0, .10, .20, .30, .40, .50 },
                new double[] { .40, .50, .60, .70, .80, .90 }
            };

            var actual = values.Exp(2);
            var expected = new double[] { 1.2214027581601699, 1.8221188003905089 };
            
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        public void ArrayExtensions_CumSum()
        {
            var values = new double[] { 0, 10, 20, 30, 40, 50 };
            var actual = values.CumSum();

            var expected = new double[] { 0, 10, 30, 60, 100, 150 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ArrayExtensions_Sum_Indexed()
        {
            var values = new double[] { 0, 10, 20, 30, 40, 50 };
            var indices = new int[] { 0, 2, 3 };
            var actual = values.Sum(indices);

            Assert.AreEqual(50, actual);
        }

        [TestMethod]
        public void ArrayExtensions_WeightedMean()
        {
            var values = new double[] { 0, 10, 20, 30, 40, 50 };
            var weights = new double[] { 1.0, 1.5, 0.1, 3, 0.2, 0.1 };
            var actual = values.WeightedMean(weights);

            Assert.AreEqual(20.33898305084746, actual, 0.0001);
        }

        [TestMethod]
        public void ArrayExtensions_WeightedMean_Indexed()
        {
            var values = new double[] { 0, 10, 20, 30, 40, 50 };
            var weights = new double[] { 1.0, 1.5, 0.1, 3, 0.2, 0.1 };
            var indices = new int[] { 0, 2, 3 };
            var actual = values.WeightedMean(weights, indices);

            Assert.AreEqual(22.439024390243905, actual, 0.0001);
        }

        [TestMethod]
        public void ArrayExtensions_WeightedMedian_1()
        {
            int n = 10;
            var w = new double[n];
            var x = new double[n];

            for (int j = 0; j < n; j++) {
                    w[j] = j + 1;
                    x[j] = j;
            }

            var actual = x.WeightedMedian(w);

            Assert.AreEqual(6, actual, 0.0001);
        }

        [TestMethod]
        public void ArrayExtensions_WeightedMedian_2()
        {
            int n = 9;
            var w = new double[n];
            var x = new double[n];

            for (int j = 0; j < n; j++) {
                w[j] = j + ((j<6) ? 1 : 0);
	            x[j] = j + 1;
            }

            var actual = x.WeightedMedian(w);

            Assert.AreEqual(6, actual, 0.0001);
        }

        [TestMethod]
        public void ArrayExtensions_Median()
        {
            var values = new double[] { 0, 10, 20, 30, 40, 50, 60 };
            var actual = values.Median();

            Assert.AreEqual(30, actual);
        }

        [TestMethod]
        public void ArrayExtensions_Median_Equal_Number_Of_Items()
        {
            var values = new double[] { 0, 10, 20, 30, 40, 50, 60, 70 };
            var actual = values.Median();

            Assert.AreEqual(35, actual);
        }

        [TestMethod]
        public void ArrayExtensions_Median_Indices()
        {
            var values = new double[] { 0, 10, 20, 30, 40, 50, 60 };
            var actual = values.Median(new int[] { 0, 1, 2 });

            Assert.AreEqual(10, actual);
        }

        [TestMethod]
        public void ArrayExtensions_ScoreAtPercentile_100()
        {
            var values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var actual = values.ScoreAtPercentile(1.0);

            Assert.AreEqual(10, actual);
        }

        [TestMethod]
        public void ArrayExtensions_ScoreAtPercentile_000()
        {
            var values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var actual = values.ScoreAtPercentile(0.0);

            Assert.AreEqual(1, actual);
        }

        [TestMethod]
        public void ArrayExtensions_ScoreAtPercentile_050()
        {
            var values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var actual = values.ScoreAtPercentile(.5);

            Assert.AreEqual(5.5, actual);
        }

        [TestMethod]
        public void ArrayExtensions_ScoreAtPercentile_080()
        {
            var values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var actual = values.ScoreAtPercentile(.8);

            Assert.AreEqual(8.2, actual);
        }

        [TestMethod]
        public void ArrayExtensions_ScoreAtPercentile_090()
        {
            var values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var actual = values.ScoreAtPercentile(.9);

            Assert.AreEqual(9.1, actual);
        }

        [TestMethod]
        public void ArrayExtensions_ScoreAtPercentile_010()
        {
            var values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            var actual = values.ScoreAtPercentile(.1);

            Assert.AreEqual(2.0, actual);
        }

        [TestMethod]
        public void ArrayExtensions_ScoreAtPercentile_010_Indexed()
        {
            var values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            var indices = Enumerable.Range(0, values.Length).ToArray();
            var actual = values.ScoreAtPercentile(.1, indices);

            Assert.AreEqual(2.0, actual);
        }

        [TestMethod]
        public void ArrayExtensions_ScoreAtPercentile_090_Indexed()
        {
            var values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var indices = Enumerable.Range(0, values.Length).ToArray();
            var actual = values.ScoreAtPercentile(.9, indices);

            Assert.AreEqual(9.1, actual);
        }

        [TestMethod]
        public void ArrayExtensions_ScoreAtPercentile_090_Indexed_2()
        {
            var values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            var indices = Enumerable.Range(0, values.Length - 1).ToArray();
            var actual = values.ScoreAtPercentile(.9, indices);

            Assert.AreEqual(9.1, actual);
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

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ArrayExtensions_Stratify_IndicesLength_And_valuesLength_Differs()
        {
            var indices = new int[] { 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            var values = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            indices.Stratify(values, new Random(42), 10);
        }
    }
}
