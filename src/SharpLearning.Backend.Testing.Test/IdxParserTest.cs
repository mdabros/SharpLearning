using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Backend.Testing.Test
{
    [TestClass]
    public class IdxParserTest
    {
        // TODO: Add ReadAll test

        [TestMethod]
        public void IdxParserTest_ParseHeader()
        {
            // Train images and labels
            ParseHeaderAndAssert(
                new byte[] { 0x00, 0x00, 0x08, 0x03, 0x00, 0x00, 0xEA, 0x60, 0x00, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x00, 0x1C }, 
                IdxDataType.Byte, new int[] { 60000, 28, 28 });
            ParseHeaderAndAssert(
                new byte[] { 0x00, 0x00, 0x08, 0x01, 0x00, 0x00, 0xEA, 0x60 },
                IdxDataType.Byte, new int[] { 60000 });
            // Test images and labels
            ParseHeaderAndAssert(
                new byte[] { 0x00, 0x00, 0x08, 0x03, 0x00, 0x00, 0x27, 0x10, 0x00, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x00, 0x1C },
                IdxDataType.Byte, new int[] { 10000, 28, 28 });
            ParseHeaderAndAssert(
                new byte[] { 0x00, 0x00, 0x08, 0x01, 0x00, 0x00, 0x27, 0x10 },
                IdxDataType.Byte, new int[] { 10000 });
        }

        [TestMethod]
        public void IdxParserTest_ParseFormat()
        {
            ParseFormatAndAssert(IdxDataType.Byte, 1);
            ParseFormatAndAssert(IdxDataType.Byte, 2);
            ParseFormatAndAssert(IdxDataType.Byte, 3);
            ParseFormatAndAssert(IdxDataType.Float, 3);
            ParseFormatAndAssert(IdxDataType.Float, 4);
        }

        [TestMethod]
        public void IdxParserTest_ParseSize()
        {
            ParseSizeAndAssert(new byte[] { 0x12, 0x34, 0x56, 0x78 }, 0x12345678);
            ParseSizeAndAssert(new byte[] { 0x00, 0x00, 0x27, 0x10 }, 10000);
            ParseSizeAndAssert(new byte[] { 0x00, 0x00, 0x00, 0x1C }, 28);
        }

        [TestMethod]
        public void IdxParserTest_ReadData()
        {
            ReadDataAndAssert(new byte[] { 0x12, 0x34, 0x56, 0x78 }, new byte[] { 0x12, 0x34, 0x56, 0x78 });
            // NOTE: If more that requested data we only read that and do not fail, this is intentional,
            //       allowing for batch reading.
            ReadDataAndAssert(new byte[] { 0x12, 0x34, 0x56, 0x78, 0x90 }, new byte[] { 0x12, 0x34, 0x56, 0x78 });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void IdxParserTest_ReadData_LessDataThanRequestedThrows()
        {
            ReadDataAndAssert(new byte[] { 0x12, 0x34, 0x56 }, new byte[] { 0x12, 0x34, 0x56, 0x78 });
        }

        void ParseHeaderAndAssert(byte[] bytes, IdxDataType expectedType, int[] expectedShape)
        {
            using (var s = new MemoryStream(bytes))
            {
                var (type, shape) = IdxParser.ParseHeader(s);

                Assert.AreEqual(expectedType, type);
                CollectionAssert.AreEqual(expectedShape, shape);
            }
        }

        static void ParseFormatAndAssert(IdxDataType expectedType, int expectedDims)
        {
            using (var s = new MemoryStream(new byte[] { 0x00, 0x00, (byte)expectedType, (byte)expectedDims }))
            {
                var (type, dims) = IdxParser.ParseFormat(s);
                Assert.AreEqual(expectedType, type);
                Assert.AreEqual(expectedDims, dims);
            }
        }

        static void ParseSizeAndAssert(byte[] bytes, int expectedSize)
        {
            using (var s = new MemoryStream(bytes))
            {
                var size = IdxParser.ParseSize(s);
                Assert.AreEqual(expectedSize, size);
            }
        }

        void ReadDataAndAssert(byte[] bytes, byte[] expectedData)
        {
            using (var s = new MemoryStream(bytes))
            {
                var data = new byte[expectedData.Length];
                IdxParser.ReadData(s, data);
                CollectionAssert.AreEqual(expectedData, data);
            }
        }
    }
}
