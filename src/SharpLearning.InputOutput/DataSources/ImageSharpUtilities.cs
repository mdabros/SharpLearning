using System.Runtime.InteropServices;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpLearning.InputOutput.DataSources
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
        /// <param name="imageGetter"></param>
        /// <returns></returns>
        public static byte[] ToBytes<TPixel>(ImageGetter<TPixel> imageGetter)
            where TPixel : struct, IPixel<TPixel>
        {
            using (var image = imageGetter())
            {
                var bytes = MemoryMarshal.AsBytes(image.GetPixelSpan()).ToArray();
                return bytes;
            }
        }
    }
}
