using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Backend.Testing.Test
{
    [TestClass]
    public class BitOpsTest
    {
        [TestMethod]
        public void BitOpsTest_BigEndianToInt32()
        {
            if (BitConverter.IsLittleEndian)
            {
                Assert.AreEqual(0x78563412, BitOps.BigEndianToInt32(0x12345678));
            }
            else
            {
                Assert.AreEqual(0x12345678, BitOps.BigEndianToInt32(0x12345678));
            }
        }

        [TestMethod]
        public void BitOpsTest_EndianSwapBytes_ushort()
        {
            Assert.AreEqual((ushort)0x3412u, BitOps.EndianSwapBytes((ushort)0x1234u));
        }

        [TestMethod]
        public void BitOpsTest_EndianSwapBytes_uint()
        {
            Assert.AreEqual(0x78563412u, BitOps.EndianSwapBytes(0x12345678u));
        }

        [TestMethod]
        public void BitOpsTest_EndianSwapBytes_ulong()
        {
            Assert.AreEqual(0xEFCDAB8967452301ul, BitOps.EndianSwapBytes(0x0123456789ABCDEFul));
        }
    }
}
