using System;
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
        /// <param name="degrees"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Rotate<TPixel>(this ImageGetter<TPixel> imageGetter, float maxDegrees, Random random)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Transform()
            {
                var degrees = (float)random.NextDouble() * maxDegrees;
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
