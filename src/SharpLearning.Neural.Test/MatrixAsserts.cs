using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Neural.Test;

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
        var m1Array = expected.ToRowMajorArray();
        var m2Array = actual.ToRowMajorArray();

        Assert.AreEqual(m1Array.Length, m2Array.Length);

        for (var i = 0; i < m1Array.Length; i++)
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

        for (var i = 0; i < expected.Length; i++)
        {
            Assert.AreEqual(expected[i], actual[i], delta);
        }
    }
}
