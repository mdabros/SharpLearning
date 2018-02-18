using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Optimization.Test.OptimizerParameters
{
    [TestClass]
    public class RandomUniformTest
    {
        [TestMethod]
        public void RandomUniform_Linear()
        {
            var random = new Random(32);
            var expected = new double[] { 99.8935983236384, 57.2098020451189, 44.4149092419142, 89.9002946307418, 137.643828772774, 114.250629522954, 63.8914499915631, 109.294177409864, 188.567149950455, 33.2731248034505 };
            RunAndAssert(expected, () => RandomUniform.Linear(min: 20, max: 200, random: random));
        }

        [TestMethod]
        public void RandomUniform_Logarithmic()
        {
            var random = new Random(32);
            var expected = new double[] { 0.00596229274859676, 0.000671250295495889, 0.000348781578382963, 0.00357552550811494, 0.0411440752926137, 0.012429636665806, 0.000944855847942692, 0.00964528475124291, 0.557104498829374, 0.000197223348905772, };
            RunAndAssert(expected, () => RandomUniform.Logarithmic(min: 0.0001, max: 1, random: random));
        }

        static void RunAndAssert(double[] expected, Func<double> linear)
        {
            var actual = new double[expected.Length];
            for (int i = 0; i < expected.Length; i++)
            {
                actual[i] = linear();
            }

            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }
    }
}
