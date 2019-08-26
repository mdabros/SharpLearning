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
        /// Rotate the image.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="maxDegrees"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Rotate<TPixel>(this ImageGetter<TPixel> imageGetter, float maxDegrees, Random random)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Transform()
            {
                var degrees = random.Sample(maxDegrees);
                var image = imageGetter();
                image.Mutate(x => x.Rotate(degrees));
                return image;
            }
            return () => Transform();
        }

        /// <summary>
        /// Alter the brightness component of the image.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="minAmount"></param>
        /// <param name="maxAmount"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Brightness<TPixel>(this ImageGetter<TPixel> imageGetter, float minAmount, float maxAmount, Random random)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Transform()
            {
                var amount = random.Sample(minAmount, maxAmount);
                var image = imageGetter();
                image.Mutate(x => x.Brightness(amount));
                return image;
            }
            return () => Transform();
        }

        /// <summary>
        /// Flip image, in selected mode, randomly.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="flipMode"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Flip<TPixel>(this ImageGetter<TPixel> imageGetter, FlipMode flipMode, Random random)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Transform()
            {
                var flip = random.NextDouble() > 0.5;
                var image = imageGetter();
                if(flip)
                {
                    image.Mutate(x => x.Flip(flipMode));
                }
                
                return image;
            }
            return () => Transform();
        }

        /// <summary>
        /// Alter skew of the image.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="maxDegreesX"></param>
        /// <param name="maxDegreesY"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Skew<TPixel>(this ImageGetter<TPixel> imageGetter, float maxDegreesX, float maxDegreesY, Random random)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Transform()
            {
                var degreesX = random.Sample(maxDegreesX);
                var degreesY = random.Sample(maxDegreesY);

                var image = imageGetter();
                image.Mutate(x => x.Skew(degreesX, degreesY));

                return image;
            }
            return () => Transform();
        }

        /// <summary>
        /// Zoom to alter the image. 
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="maxZoom">minimum value is 1.0</param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Zoom<TPixel>(this ImageGetter<TPixel> imageGetter, float maxZoom, Random random)
            where TPixel : struct, IPixel<TPixel>
        {
            if (maxZoom < 1) throw new ArgumentException("Zoom must be at least 1.0");

            Image<TPixel> Transform()
            {
                var zoom = random.Sample(1, maxZoom);
                var image = imageGetter();

                var width = image.Width;
                var height = image.Height;

                // trunk 
                var resizeWidth = (int)(width * zoom);
                var resizeHeight = (int)(height * zoom);

                // enlarge and crop to zoom.
                // TODO: Add random topleft for crop (currently always 0.0).
                image.Mutate(x => x.Resize(resizeWidth, resizeHeight).Crop(width, height));
                return image;
            }
            return () => Transform();
        }

        /// <summary>
        /// Custom operation.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="operation"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Apply<TPixel>(this ImageGetter<TPixel> imageGetter, Action<Image<TPixel>> operation)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Transform()
            {
                var image = imageGetter();
                image.Mutate(x => x.Apply(operation));
                return image;
            }
            return () => Transform();
        }

        // TODO: Add remaining operators:

        //public static void Flip<TPixel>(Image<TPixel> image, FlipMode flipMode)
        //    where TPixel : struct, IPixel<TPixel> => image.Mutate(x => x.Flip(flipMode));

    }
}
