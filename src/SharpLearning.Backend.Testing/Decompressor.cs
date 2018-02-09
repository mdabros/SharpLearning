using System.IO;
using System.IO.Compression;

namespace SharpLearning.Backend.Testing
{
    public static class Decompressor
    {
        public static Stream Decompress(this Stream input) => new GZipStream(input, CompressionMode.Decompress);
    }
}
