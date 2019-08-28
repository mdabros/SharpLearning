using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace SharpLearning.InputOutput.DataSources
{
    /// <summary>
    /// Methods for handling images via ImageSharp.
    /// </summary>
    public static partial class ImageSharpUtilities
    {
        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Resize<TPixel>(this ImageGetter<TPixel> imageGetter, int width, int height)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Transform()
            {
                var image = imageGetter();
                image.Mutate(x => x.Resize(width, height));
                return image;
            }
            return () => Transform();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Pad<TPixel>(this ImageGetter<TPixel> imageGetter, int width, int height)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Transform()
            {
                var image = imageGetter();
                image.Mutate(x => x.Pad(width, height));
                return image;
            }
            return () => Transform();
        }
    }
}
