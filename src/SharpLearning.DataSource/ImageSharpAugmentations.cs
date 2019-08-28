using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.Primitives;

namespace SharpLearning.DataSource
{
    /// <summary>
    /// Image augmentation methods using ImageSharp.
    /// </summary>
    public static class ImageSharpAugmentations
    {
        /// <summary>
        /// Rotate the image.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="maxDegrees"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Rotate<TPixel>(this ImageGetter<TPixel> imageGetter,
            float maxDegrees, Random random)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Augment()
            {
                var degrees = random.SampleUniform(0, maxDegrees);
                var image = imageGetter();
                image.Mutate(x => x.Rotate(degrees));
                return image;
            }
            return () => Augment();
        }

        /// <summary>
        /// Flip image, in selected mode, randomly.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="flipMode"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Flip<TPixel>(this ImageGetter<TPixel> imageGetter,
            FlipMode flipMode, Random random)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Augment()
            {
                var flip = random.NextDouble() > 0.5;
                var image = imageGetter();
                if (flip)
                {
                    image.Mutate(x => x.Flip(flipMode));
                }

                return image;
            }
            return () => Augment();
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
        public static ImageGetter<TPixel> Skew<TPixel>(this ImageGetter<TPixel> imageGetter,
            float maxDegreesX, float maxDegreesY, Random random)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Augment()
            {
                var degreesX = random.SampleUniform(0, maxDegreesX);
                var degreesY = random.SampleUniform(0, maxDegreesY);

                var image = imageGetter();
                image.Mutate(x => x.Skew(degreesX, degreesY));

                return image;
            }
            return () => Augment();
        }

        /// <summary>
        /// Zoom to alter the image. 
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="maxZoom">minimum value is 1.0</param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Zoom<TPixel>(this ImageGetter<TPixel> imageGetter,
            float maxZoom, Random random)
            where TPixel : struct, IPixel<TPixel>
        {
            if (maxZoom < 1) throw new ArgumentException("Zoom must be at least 1.0");

            Image<TPixel> Augment()
            {
                var image = imageGetter();
                var width = image.Width;
                var height = image.Height;

                // Enlarge width/height, keep ratio.
                var scale = random.SampleUniform(1, maxZoom);
                var resizeWidth = (int)Math.Round(width * scale);
                var resizeHeight = (int)Math.Round(height * scale);

                // Find random crop location to get original size.
                var maxX = resizeWidth - width;
                var maxY = resizeHeight - height;
                var x = (int)Math.Round(random.SampleUniform(0, maxX));
                var y = (int)Math.Round(random.SampleUniform(0, maxY));
                var cropRectangle = new Rectangle(x, y, width, height);

                // Enlarge and crop to zoom.                
                image.Mutate(v => v.Resize(resizeWidth, resizeHeight)
                    .Crop(cropRectangle));

                return image;
            }
            return () => Augment();
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
        public static ImageGetter<TPixel> Brightness<TPixel>(this ImageGetter<TPixel> imageGetter,
            float minAmount, float maxAmount, Random random)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Augment()
            {
                var amount = random.SampleUniform(minAmount, maxAmount);
                var image = imageGetter();
                image.Mutate(x => x.Brightness(amount));
                return image;
            }
            return () => Augment();
        }

        /// <summary>
        /// Custom augmentation operation.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageGetter"></param>
        /// <param name="operation"></param>
        /// <returns></returns>
        public static ImageGetter<TPixel> Apply<TPixel>(this ImageGetter<TPixel> imageGetter,
            Action<Image<TPixel>> operation)
            where TPixel : struct, IPixel<TPixel>
        {
            Image<TPixel> Augment()
            {
                var image = imageGetter();
                image.Mutate(x => x.Apply(operation));
                return image;
            }
            return () => Augment();
        }
    }
}
