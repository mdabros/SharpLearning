using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.DataSource.Test
{
    public static class AssertUtilities
    {
        const double m_defaultDelta = 0.000001;

        public static void AssertAreEqual(
            IReadOnlyDictionary<string, DataBatch<double>> expected,
            IReadOnlyDictionary<string, DataBatch<double>> actual,
            double delta = m_defaultDelta)
        {
            Assert.AreEqual(expected.Count, actual.Count);
            foreach (var key in expected.Keys)
            {
                Assert.IsTrue(actual.ContainsKey(key));
                AssertAreEqual(expected[key], actual[key], delta);
            }
        }

        public static void AssertAreEqual(DataBatch<double> expected, DataBatch<double> actual, 
            double delta = m_defaultDelta)
        {
            Assert.AreEqual(expected.SampleCount, actual.SampleCount);
            CollectionAssert.AreEqual(expected.SampleShape, actual.SampleShape);

            Assert.AreEqual(expected.Data.Length, actual.Data.Length);
            for (int i = 0; i < expected.Data.Length; i++)
            {
                Assert.AreEqual(expected.Data[i], actual.Data[i], delta);
            }
        }
    }
}
