using System;
using System.Globalization;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Containers.Test;

[TestClass]
public class FloatingPointConversionTest
{
    [TestMethod]
    public void FloatingPointConversion_ToString()
    {
        var value = 1.2354236;
        var actual = FloatingPointConversion.ToString(value);
        Assert.AreEqual("1.2354236", actual);
    }

    [TestMethod]
    public void FloatingPointConversion_ToF64()
    {
        var value = "1.2354236";
        var actual = ToF64(value);
        Assert.AreEqual(1.2354236, actual);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void FloatingPointConversion_ToF64_Unable_To_Parse()
    {
        ToF64("infinity12");
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void FloatingPointConversion_ToF64_win10_infinity_symbol()
    {
        // https://stackoverflow.com/questions/40907417/why-is-infinity-printed-as-8-in-the-windows-10-console
        var win10Infinity = "∞";
        Assert.AreEqual(double.PositiveInfinity, ToF64(win10Infinity));
    }

    [TestMethod]
    public void FloatingPointConversion_ToF64_to_from_infinity()
    {
        var posNegInfinity = new double[] { 1.0 / 0.0, -1.0 / 0.0 };
        var text = posNegInfinity.Select(FloatingPointConversion.ToString).ToArray();
        var actual = text.Select(ToF64).ToArray();
        var expected = new double[] { double.PositiveInfinity, double.NegativeInfinity };
        CollectionAssert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void FloatingPointConversion_ToF64_to_from_custom_infinity()
    {
        var text = new string[] { "Inf", "beyond", "0.0", "75357.987" };

        var nfi = new NumberFormatInfo()
        {
            PositiveInfinitySymbol = text[0],
            NegativeInfinitySymbol = text[1],
        };

        var actual = text.Select(x => FloatingPointConversion.ToF64(x,
            converter: t => double.Parse(t, nfi))).ToArray();

        var expected = new double[] { double.PositiveInfinity, double.NegativeInfinity, 0.0, 75357.987 };
        CollectionAssert.AreEqual(expected, actual);
    }

    static double ToF64(string value)
    {
        return FloatingPointConversion.ToF64(value);
    }
}
