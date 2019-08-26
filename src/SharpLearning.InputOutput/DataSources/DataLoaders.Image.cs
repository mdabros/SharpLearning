using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpLearning.InputOutput.Csv;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpLearning.InputOutput.DataSources
{
    public static partial class DataLoaders
    {
        /// <summary>
        /// Load data from images.
        /// </summary>
        public static class Image
        {
            /// <summary>
            /// Load images from csv image name columns, and image directory path.
            /// </summary>
            /// <param name="csvText"></param>
            /// <param name="imageNameColumnName"></param>
            /// <param name="imagesDirectoryPath"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<float> FromCsvText<TPixel>(string csvText,
                string imageNameColumnName,
                string imagesDirectoryPath,
                int[] sampleShape) where TPixel : struct, IPixel<TPixel>
            {
                var parser = CsvParser.FromText(csvText);
                return FromCsvParser<TPixel>(parser, imageNameColumnName, imagesDirectoryPath, sampleShape);
            }

            /// <summary>
            /// Load images from csv image name columns, and image directory path.
            /// </summary>
            /// <param name="csvFilePath"></param>
            /// <param name="imageNameColumnName"></param>
            /// <param name="imagesDirectoryPath"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<float> FromCsvFile<TPixel>(string csvFilePath,
                string imageNameColumnName,
                string imagesDirectoryPath,
                int[] sampleShape) where TPixel : struct, IPixel<TPixel>
            {
                var parser = CsvParser.FromFile(csvFilePath);
                return FromCsvParser<TPixel>(parser, imageNameColumnName, imagesDirectoryPath, sampleShape);
            }

            /// <summary>
            /// Load images from csv image name columns, and image directory path.
            /// </summary>
            /// <param name="parser"></param>
            /// <param name="imageNameColumnName"></param>
            /// <param name="imagesDirectoryPath"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<float> FromCsvParser<TPixel>(CsvParser parser,
                string imageNameColumnName,
                string imagesDirectoryPath,
                int[] sampleShape) where TPixel : struct, IPixel<TPixel>
            {
                var imageFilePaths = parser.EnumerateRows(imageNameColumnName)
                    .ToStringVector().Select(filename => Path.Combine(imagesDirectoryPath, filename))
                    .ToArray();

                var imageGetters = ImageUtilities.EnumerateImages<TPixel>(imageFilePaths);
                return FromImageGetters(imageGetters, sampleShape);
            }

            /// <summary>
            /// Load images from image file paths.
            /// </summary>
            /// <param name="imageFilePaths"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<float> FromImageGetters<TPixel>(
                IEnumerable<ImageGetter<TPixel>> imageGetters,
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
}
