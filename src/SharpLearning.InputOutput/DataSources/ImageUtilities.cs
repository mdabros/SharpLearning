using System;
using System.Linq;
using System.Runtime.InteropServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpLearning.InputOutput.DataSources
{
    /// <summary>
    /// Image transform delegate.
    /// </summary>
    /// <typeparam name="TPixel"></typeparam>
    /// <param name="imageGetter"></param>
    /// <returns></returns>
    public delegate Func<Image<TPixel>> ImageTransformer<TPixel>(Func<Image<TPixel>> imageGetter)
        where TPixel : struct, IPixel<TPixel>;

    /// <summary>
    /// Methods for handling images via ImageSharp.
    /// </summary>
    public static partial class ImageUtilities
    {
        /// <summary>
        /// Gets func for loading image.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public static Func<Image<TPixel>> GetImageLoader<TPixel>(string filePath) 
            where TPixel : struct, IPixel<TPixel>
        {
            return () => Image.Load<TPixel>(filePath);
        }

        /// <summary>
        /// Combines transforms.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageLoader"></param>
        /// <param name="imageTransforms"></param>
        /// <returns></returns>
        public static Func<Image<TPixel>> GetImageTransformer<TPixel>(Func<Image<TPixel>> imageLoader,
            ImageTransformer<TPixel>[] imageTransforms) where TPixel : struct, IPixel<TPixel>
        {
            Func<Image<TPixel>> imageTransformer = imageLoader;
            foreach (var transform in imageTransforms)
            {
                imageTransformer = transform(imageTransformer);
            }

            return imageTransformer;
        }

        /// <summary>
        /// Converts image to byte array.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <returns></returns>
        public static byte[] ConvertImageToBytes<TPixel>(Func<Image<TPixel>> imageGetter)
            where TPixel : struct, IPixel<TPixel>
        {
            using (var image = imageGetter())
            {
                var bytes = MemoryMarshal.AsBytes(image.GetPixelSpan()).ToArray();
                return bytes;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="bytes"></param>
        /// <returns></returns>
        public static float[] ConvertBytesToFloat(byte[] bytes)
        {
            return bytes.Select(v => (float)v).ToArray();
        }
    }
}
