using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Views;

namespace SharpLearning.Containers.Test.Views
{
    [TestClass]
    public class F64VectorViewTest
    {
        [TestMethod]
        public void F64VectorView_Index()
        {
            var vector = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            using (var pinned = vector.GetPinnedPointer())
            {
                var view = pinned.View();
                AssertVectorView(vector, view);
            }
        }

        [TestMethod]
        public void F64VectorView_SubView_Start()
        {
            var vector = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            using (var pinned = vector.GetPinnedPointer())
            {
                var view = pinned.View().View(Interval1D.Create(0, 3));
                var expected = new double[] { 0, 1, 2 };
                AssertVectorView(expected, view);
            }
        }

        [TestMethod]
        public void F64VectorView_SubView_Middle()
        {
            var vector = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            using (var pinned = vector.GetPinnedPointer())
            {
                var view = pinned.View().View(Interval1D.Create(3, 6));
                var expected = new double[] { 3, 4, 5 };
                AssertVectorView(expected, view);
            }
        }

        [TestMethod]
        public void F64VectorView_SubView_End()
        {
            var vector = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            using (var pinned = vector.GetPinnedPointer())
            {
                var view = pinned.View().View(Interval1D.Create(7, 10));
                var expected = new double[] { 7, 8, 9 };
                AssertVectorView(expected, view);
            }
        }

        void AssertVectorView(double[] vector, F64VectorView view)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                Assert.AreEqual(vector[i], view[i]);
            }
        }
    }
}
