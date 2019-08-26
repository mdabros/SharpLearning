using System;
using System.Collections.Generic;
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
    public delegate ImageGetter<TPixel> ImageTransformer<TPixel>(ImageGetter<TPixel> imageGetter)
        where TPixel : struct, IPixel<TPixel>;

    /// <summary>
    /// Image getter delegate
    /// </summary>
    /// <typeparam name="TPixel"></typeparam>
    /// <returns></returns>
    public delegate Image<TPixel> ImageGetter<TPixel>()
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
        public static ImageGetter<TPixel> GetImageLoader<TPixel>(string filePath) 
            where TPixel : struct, IPixel<TPixel>
        {
            return () => Image.Load<TPixel>(filePath);
        }

        /// <summary>
        /// Combines transforms.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="imageTransforms"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> GetImageTransformer<TPixel>(ImageGetter<TPixel> imageGetter,
            ImageTransformer<TPixel>[] imageTransforms) where TPixel : struct, IPixel<TPixel>
        {
            ImageGetter<TPixel> imageTransformer = imageGetter;
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
        public static byte[] ConvertImageToBytes<TPixel>(ImageGetter<TPixel> imageGetter)
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

        /// <summary>
        /// Enumerates a list of images filepaths.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageFilePaths"></param>
        /// <returns></returns>
        public static IEnumerable<ImageGetter<TPixel>> EnumerateImages<TPixel>(string[] imageFilePaths)
             where TPixel : struct, IPixel<TPixel>
        {
            foreach (var imageFilePath in imageFilePaths)
            {
                yield return GetImageLoader<TPixel>(imageFilePath);
            }
        }
    }
}
