using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using System.Linq;

namespace SharpLearning.DecisionTrees.Test.ImpurityCalculators
{
    [TestClass]
    public class GiniClassificationImpurityCalculatorTest
    {
        [TestMethod]
        public void GiniClassificationImpurityCalculator_ImpurityImprovement()
        {
            var values = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, };
            var unique = values.Distinct().ToArray();

            var parentInterval = Interval1D.Create(0, values.Length);

            var sut = new GiniClassificationImpurityCalculator();
            sut.Init(unique, values, new double[0], parentInterval);

            var impurity = sut.NodeImpurity();

            sut.UpdateIndex(50);
            var improvement1 = sut.ImpurityImprovement(impurity);
            Assert.AreEqual(0.33333333333333343, improvement1, 0.000001);

            sut.UpdateIndex(96);
            var improvement2 = sut.ImpurityImprovement(impurity);
            Assert.AreEqual(0.28047839506172845, improvement2, 0.000001);
        }

        [TestMethod]
        public void GiniClassificationImpurityCalculator_ImpurityImprovement_Weighted()
        {
            var values = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, };
            var unique = values.Distinct().ToArray();

            var weights = values.Select(t => Weight(t)).ToArray();
            var parentInterval = Interval1D.Create(0, values.Length);

            var sut = new GiniClassificationImpurityCalculator();
            sut.Init(unique, values, weights, parentInterval);

            var impurity = sut.NodeImpurity();

            sut.UpdateIndex(50);
            var improvement1 = sut.ImpurityImprovement(impurity);
            Assert.AreEqual(0.14015151515151511, improvement1, 0.000001);

            sut.UpdateIndex(96);
            var improvement2 = sut.ImpurityImprovement(impurity);
            Assert.AreEqual(0.17358104858104859, improvement2, 0.000001);
        }

        [TestMethod]
        public void GiniClassificationImpurityCalculator_ChildImpurities()
        {
            var values = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, };
            var unique = values.Distinct().ToArray();

            var parentInterval = Interval1D.Create(0, values.Length);

            var sut = new GiniClassificationImpurityCalculator();
            sut.Init(unique, values, new double[0], parentInterval);

            var impurity = sut.NodeImpurity();

            sut.UpdateIndex(50);
            var actual = sut.ChildImpurities();
            var expected = new ChildImpurities(0.0, .5);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void GiniClassificationImpurityCalculator_NodeImpurity()
        {
            var values = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, };
            var unique = values.Distinct().ToArray();

            var parentInterval = Interval1D.Create(0, values.Length);

            var sut = new GiniClassificationImpurityCalculator();
            sut.Init(unique, values, new double[0], parentInterval);

            sut.UpdateIndex(50);
            var actual = sut.NodeImpurity();

            Assert.AreEqual(0.66666666666666674, actual, 0.000001);
        }

        [TestMethod]
        public void GiniClassificationImpurityCalculator_LeafValue_Weighted()
        {
            var values = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, };
            var unique = values.Distinct().ToArray();
            var weights = values.Select(t => Weight(t)).ToArray();
            var parentInterval = Interval1D.Create(0, values.Length);

            var sut = new GiniClassificationImpurityCalculator();
            sut.Init(unique, values, weights, parentInterval);

            var impurity = sut.NodeImpurity();

            sut.UpdateIndex(50);
            var actual = sut.LeafValue();

            Assert.AreEqual(2.0, actual, 0.000001);
        }

        double Weight(double t)
        {
            if (t == 2.0)
                return 10.0;
            return 1.0;
        }
    }
}
