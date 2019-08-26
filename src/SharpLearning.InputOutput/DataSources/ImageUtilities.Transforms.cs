using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace SharpLearning.InputOutput.DataSources
{
    /// <summary>
    /// Methods for handling images via ImageSharp.
    /// </summary>
    public static partial class ImageUtilities
    {
        /// <summary>
        /// Image transforms
        /// </summary>
        public static class Transforms
        {
            /// <summary>
            /// 
            /// </summary>
            /// <typeparam name="TPixel"></typeparam>
            /// <param name="imageGetter"></param>
            /// <param name="width"></param>
            /// <param name="height"></param>
            /// <returns></returns>
            public static ImageGetter<TPixel> Resize<TPixel>(ImageGetter<TPixel> imageGetter, int width, int height)
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
            /// <param name="degrees"></param>
            /// <returns></returns>
            public static ImageGetter<TPixel> Rotate<TPixel>(ImageGetter<TPixel> imageGetter, float degrees)
                where TPixel : struct, IPixel<TPixel>
            {
                Image<TPixel> Transform()
                {
                    var image = imageGetter();
                    image.Mutate(x => x.Rotate(degrees));
                    return image;
                }
                return () => Transform();
            }
            
            // TODO: Add remaining operators:

            //public static void Flip<TPixel>(Image<TPixel> image, FlipMode flipMode)
            //    where TPixel : struct, IPixel<TPixel> => image.Mutate(x => x.Flip(flipMode));
        }
    }
}
