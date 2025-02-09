using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Optimization.Test;

public static class ArrayAssert
{
    const double m_defaultDelta = 0.000001;

    public static void AssertAreEqual(double[] expected, double[] actual,
        double delta = m_defaultDelta)
    {
        Assert.AreEqual(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.AreEqual(expected[i], actual[i], delta);
        }
    }
}
