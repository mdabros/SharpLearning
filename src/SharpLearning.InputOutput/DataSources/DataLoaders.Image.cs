using System;
using System.Collections.Generic;
using System.Linq;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpLearning.InputOutput.DataSources
{
    public static partial class DataLoaders
    {
        /// <summary>
        /// Creates DataLoader from ImageGetters.
        /// </summary>
        /// <param name="imageGetters"></param>
        /// <param name="sampleShape"></param>
        /// <returns></returns>
        public static DataLoader<float> ToDataLoader<TPixel>(
            this IEnumerable<ImageGetter<TPixel>> imageGetters,
            int[] sampleShape) where TPixel : struct, IPixel<TPixel>
        {
            var sampleSize = sampleShape.Aggregate((v1, v2) => v1 * v2);

            DataBatch<float> LoadImageData(int[] indices)
            {
                var batchSampleCount = indices.Length;
                var data = new float[batchSampleCount * sampleSize];
                var currentIndex = 0;
                var copyIndexStart = 0;

                foreach (var imageGetter in imageGetters)
                {
                    if (indices.Contains(currentIndex))
                    {
                        // TODO: Add conversion func.to select byte, float, double, etc.
                        // TODO: Consider augmentations.
                        var bytes = ImageUtilities.ConvertImageToBytes(imageGetter);
                        var floats = ImageUtilities.ConvertBytesToFloat(bytes);

                        Array.Copy(floats, 0, data, copyIndexStart, sampleSize);

                        copyIndexStart += sampleSize;
                    }
                    currentIndex++;
                }

                return new DataBatch<float>(data, sampleShape, batchSampleCount);
            }

            var totalSampleCount = imageGetters.Count();
            return new DataLoader<float>(LoadImageData, totalSampleCount);
        }
    }
}
