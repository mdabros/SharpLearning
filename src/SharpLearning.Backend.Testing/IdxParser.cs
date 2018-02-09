using System;
using System.IO;

namespace SharpLearning.Backend.Testing
{
    // http://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
    // http://yann.lecun.com/exdb/mnist/
    //
    // The basic format according to http://yann.lecun.com/exdb/mnist/ is:
    // ```
    // magic number
    // size in dimension 1
    // size in dimension 2
    // size in dimension 3
    // ....
    // size in dimension N
    // data
    // ```
    // The magic number is four bytes long. The first 2 bytes are always 0.
    // 
    // The third byte codes the type of the data:
    // ```
    // 0x08: unsigned byte
    // 0x09: signed byte
    // 0x0B: short (2 bytes)
    // 0x0C: int (4 bytes)
    // 0x0D: float (4 bytes)
    // 0x0E: double (8 bytes)
    // ```
    // The fouth byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
    // 
    // The sizes in each dimension are 4-byte integers(big endian, like in most non-Intel processors).
    // 
    // The data is stored like in a C array, i.e.the index in the last dimension changes the fastest.
    //
    // Behaviour
    // If the storage format indicates that there are more than 2 dimensions, 
    // the resulting Matrix accumulates dimensions 2 and higher in the columns.
    // For example, with three dimensions of size n1, n2 and n3, respectively, 
    // the resulting Matrix object will have n1 rows and n2×n3 columns.
    // 
    // Example
    // The training and testing data of the MNIST database of handwritten digits at 
    // http://yann.lecun.com/exdb/mnist/ is stored in compressed IDX formatted files.
    // 
    // Reading the uncompressed file train-images-idx3-ubyte available at http://yann.lecun.com/exdb/mnist/ 
    // with 60000 images of 28×28 pixel data, will result in a new Matrix object with 60000 rows and 784 (=28×28) columns. 
    // Each cell will contain a number in the interval from 0 to 255.
    // 
    // Reading the uncompressed file train-labels-idx1-ubyte with 60000 labels will result in 
    // a new Matrix object with 1 row and 60000 columns.Each cell will contain a number in the interval from 0 to 9.
    public static class IdxParser
    {
        public static (IdxDataType type, int[] shape) ParseHeader(Stream s)
        {
            var (type, dimensions) = ParseFormat(s);
            var shape = new int[dimensions];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = ParseSize(s);
            }
            return (type, shape);
        }

        public static (IdxDataType type, int dimensions) ParseFormat(Stream s)
        {
            var i = ReadBigEndianInt32(s);
            var type = (IdxDataType)((i & 0x0000FF00) >> 8);
            var dimensions = (i & 0x000000FF);
            return (type, dimensions);
        }

        public static int ParseSize(Stream s)
        {
            return ReadBigEndianInt32(s);
        }

        //public static int ParseSize1(Stream s)
        //{
        //    return ReadBigEndianInt32(s);
        //}

        //public static (int count, int rows, int cols) ParseIdx3Header(Stream s)
        //{
        //    const int MagicNumber = 0x00000803; // 2051;
        //    ReadAndCheckMagicNumber(s, MagicNumber);
        //    var count = ReadBigEndianInt32(s);
        //    var rows = ReadBigEndianInt32(s);
        //    var cols = ReadBigEndianInt32(s);
        //    return (count, rows, cols);
        //}

        //public static void ReadAndCheckMagicNumber(Stream s, int expectedMagicNumber)
        //{
        //    var magicNumber = ReadBigEndianInt32(s);
        //    if (magicNumber != expectedMagicNumber)
        //    { throw new ArgumentException($"Invalid magic number found '{magicNumber}'"); }
        //}

        public static int ReadBigEndianInt32(Stream s)
        {
            int bigEndian = s.ReadInt32();
            return BitOps.BigEndianToInt32(bigEndian);// DataConverter.BigEndian.GetInt32(x, 0);
        }
    }

    public enum IdxDataType : byte
    {
        Byte   = 0x08, //: unsigned byte 
        SByte  = 0x09, //: signed byte 
        Short  = 0x0B, //: short (2 bytes) 
        Int    = 0x0C, //: int (4 bytes) 
        Float  = 0x0D, //: float (4 bytes) 
        Double = 0x0E, //: double (8 bytes)
    }
}
