using System.IO;

namespace SharpLearning.Backend.Testing
{
    public static class StreamExtensions
    {
        public static unsafe int ReadInt32(this Stream s)
        {
            byte* bytes = stackalloc byte[sizeof(int)];
            for (int i = 0; i < sizeof(int); i++)
            {
                var r = s.ReadByte();
                if (r < 0)
                { throw new EndOfStreamException(); }
                bytes[i] = (byte)r;
            }
            int v = *(int*)bytes;
            return v;
        }
    }
}
