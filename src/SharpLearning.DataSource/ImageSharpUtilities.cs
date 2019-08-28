using System.Runtime.InteropServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpLearning.DataSource
{
    /// <summary>
    /// Methods for handling images via ImageSharp.
    /// </summary>
    public static partial class ImageSharpUtilities
    {
        /// <summary>
        /// Converts image to byte array.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="image"></param>
        /// <returns></returns>
        public static byte[] ToBytes<TPixel>(this Image<TPixel> image) where TPixel : struct, IPixel<TPixel>
            => MemoryMarshal.AsBytes(image.GetPixelSpan()).ToArray();
    }
}
