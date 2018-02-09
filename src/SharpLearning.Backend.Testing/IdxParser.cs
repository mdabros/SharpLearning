using System;
using System.IO;

namespace SharpLearning.Backend.Testing
{
    public static class IdxParser
    {
        const int MagicNumber = 2051;

        public static int ParseIdx1Header(Stream s)
        {
            ReadAndCheckMagicNumber(s);
            var count = ReadBigEndianInt32(s);
            return count;
        }

        public static (int count, int rows, int cols) ParseIdx3Header(Stream s)
        {
            ReadAndCheckMagicNumber(s);
            var count = ReadBigEndianInt32(s);
            var rows = ReadBigEndianInt32(s);
            var cols = ReadBigEndianInt32(s);
            return (count, rows, cols);
        }

        public static void ReadAndCheckMagicNumber(Stream s)
        {
            var magicNumber = ReadBigEndianInt32(s);
            if (magicNumber != MagicNumber)
            { throw new ArgumentException($"Invalid magic number found '{magicNumber}'"); }
        }

        public static int ReadBigEndianInt32(Stream s)
        {
            int bigEndian = s.ReadInt32();
            return BitOps.BigEndianToInt32(bigEndian);// DataConverter.BigEndian.GetInt32(x, 0);
        }
    }
}
