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
    }
}
