using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Containers.Test
{
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
            var actual = FloatingPointConversion.ToF64(value);
            Assert.AreEqual(1.2354236, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FloatingPointConversion_ToF64_Unable_To_Parse()
        {
            var value = "infinity12";
            FloatingPointConversion.ToF64(value);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FloatingPointConvertsion_ToF64_win10_infinity_symbol()
        {
            // https://stackoverflow.com/questions/40907417/why-is-infinity-printed-as-8-in-the-windows-10-console
            var win10Infinity = "∞";
            Assert.AreEqual(double.PositiveInfinity, FloatingPointConversion.ToF64(win10Infinity));
        }

        [TestMethod]
        public void FloatingPointConvertsion_ToF64_infinity()
        {
            Assert.AreEqual(double.PositiveInfinity, FloatingPointConversion.ToF64("Inf"));
            Assert.AreEqual(double.NegativeInfinity, FloatingPointConversion.ToF64("-Inf"));
        }
    }
}
