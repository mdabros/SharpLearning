using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpLearning.DataSource
{
    public static partial class DataLoaders
    {
        /// <summary>
        /// Enumerates a list of image filepaths.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="imageFilePaths"></param>
        /// <returns></returns>
        public static IEnumerable<ImageGetter<TPixel>> EnumerateImages<TPixel>(IEnumerable<string> imageFilePaths)
             where TPixel : struct, IPixel<TPixel>
        {
            foreach (var imageFilePath in imageFilePaths)
            {
                yield return () => Image.Load<TPixel>(imageFilePath);
            }
        }

        /// <summary>
        /// Enumerates a list of byte array images.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="arrays"></param>
        /// <returns></returns>
        public static IEnumerable<ImageGetter<TPixel>> EnumerateImages<TPixel>(IEnumerable<byte[]> arrays)
             where TPixel : struct, IPixel<TPixel>
        {
            foreach (var array in arrays)
            {
                yield return () => Image.Load<TPixel>(array);
            }
        }

        /// <summary>
        /// Enumerates a list of stream images.
        /// </summary>
        /// <typeparam name="TPixel"></typeparam>
        /// <param name="streams"></param>
        /// <returns></returns>
        public static IEnumerable<ImageGetter<TPixel>> EnumerateImages<TPixel>(IEnumerable<Stream> streams)
             where TPixel : struct, IPixel<TPixel>
        {
            foreach (var stream in streams)
            {
                yield return () => Image.Load<TPixel>(stream);
            }
        }
        
        /// <summary>
        /// Creates DataLoader from ImageGetters.
        /// </summary>
        /// <param name="imageGetters"></param>
        /// <param name="sampleShape"></param>
        /// <returns></returns>
        public static DataLoader<TData> ToImageDataLoader<TPixel, TData>(
            this IEnumerable<ImageGetter<TPixel>> imageGetters,
            Func<byte[], TData[]> pixelConverter,
            params int[] sampleShape) where TPixel : struct, IPixel<TPixel>
        {
            var sampleSize = sampleShape.Aggregate((v1, v2) => v1 * v2);

            DataBatch<TData> LoadImageData(int[] indices)
            {
                var batchSampleCount = indices.Length;
                var data = new TData[batchSampleCount * sampleSize];
                var currentIndex = 0;
                var copyIndexStart = 0;

                foreach (var imageGetter in imageGetters)
                {
                    if (indices.Contains(currentIndex))
                    {
                        using(var image = imageGetter())
                        {
                            var bytes = image.ToBytes();
                            var imageData = pixelConverter(bytes);
                            Array.Copy(imageData, 0, data, copyIndexStart, sampleSize);
                        }
                        copyIndexStart += sampleSize;
                    }
                    currentIndex++;
                }

                return new DataBatch<TData>(data, sampleShape, batchSampleCount);
            }

            var totalSampleCount = imageGetters.Count();
            return new DataLoader<TData>(LoadImageData, totalSampleCount);
        }
    }
}
