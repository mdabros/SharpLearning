using System;
using System.Runtime.InteropServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace SharpLearning.InputOutput.DataSources
{
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

        public static byte[] ConvertImageToBytes<TPixel>(Func<Image<TPixel>> imageGetter)
            where TPixel : struct, IPixel<TPixel>
        {
            using (var image = imageGetter())
            {
                var bytes = MemoryMarshal.AsBytes(image.GetPixelSpan()).ToArray();
                return bytes;
            }
        }

        public static class Transforms
        {
            public static void Resize<TPixel>(Image<TPixel> image, int width, int height)
                where TPixel : struct, IPixel<TPixel> => image.Mutate(x => x.Resize(width, height));

            public static void Rotate<TPixel>(Image<TPixel> image, float degrees)
                where TPixel : struct, IPixel<TPixel> => image.Mutate(x => x.Rotate(degrees));

            public static void Flip<TPixel>(Image<TPixel> image, FlipMode flipMode)
                where TPixel : struct, IPixel<TPixel> => image.Mutate(x => x.Flip(flipMode));

            // TODO: Add remaining operators.
        }
    }
}
