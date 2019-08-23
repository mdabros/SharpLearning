using System;
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
            /// <param name="imageDirectoryPath"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<float> FromCsvText(string csvText,
                string imageNameColumnName,
                string imageDirectoryPath,
                int[] sampleShape)
            {
                var parser = CsvParser.FromText(csvText);
                return FromCsvParser(parser, imageNameColumnName, imageDirectoryPath, sampleShape);
            }

            /// <summary>
            /// Load images from csv image name columns, and image directory path.
            /// </summary>
            /// <param name="csvFilePath"></param>
            /// <param name="imageNameColumnName"></param>
            /// <param name="imageDirectoryPath"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<float> FromCsvFile(string csvFilePath,
                string imageNameColumnName,
                string imageDirectoryPath,
                int[] sampleShape)
            {
                var parser = CsvParser.FromFile(csvFilePath);
                return FromCsvParser(parser, imageNameColumnName, imageDirectoryPath, sampleShape);
            }

            /// <summary>
            /// Load images from csv image name columns, and image directory path.
            /// </summary>
            /// <param name="parser"></param>
            /// <param name="imageNameColumnName"></param>
            /// <param name="imageDirectoryPath"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<float> FromCsvParser(CsvParser parser,
                string imageNameColumnName,
                string imageDirectoryPath,
                int[] sampleShape)
            {
                var imageFilePaths = parser.EnumerateRows(imageNameColumnName)
                    .ToStringVector().Select(filename => Path.Combine(imageDirectoryPath, filename))
                    .ToArray();

                return FromImageFilePaths(imageFilePaths, sampleShape);
            }

            /// <summary>
            /// Load images from image file paths.
            /// </summary>
            /// <param name="imageFilePaths"></param>
            /// <param name="sampleShape"></param>
            /// <returns></returns>
            public static DataLoader<float> FromImageFilePaths(
                string[] imageFilePaths,
                int[] sampleShape)
            {
                var sampleSize = sampleShape.Aggregate((v1, v2) => v1 * v2);

                DataBatch<float> LoadImageData(int[] indices)
                {
                    var batchSampleCount = indices.Length;
                    var data = new float[batchSampleCount * sampleSize];
                    var currentIndex = 0;
                    var copyIndexStart = 0;

                    foreach (var imageFilePath in imageFilePaths)
                    {
                        if (indices.Contains(currentIndex))
                        {
                            // TODO: Add conversion func.to select byte, float, double, etc.
                            // TODO: Consider augmentations.
                            var imageGetter = ImageUtilities.GetImageLoader<Rgba32>(imageFilePath);
                            var bytes = ImageUtilities.ConvertImageToBytes(imageGetter);
                            var floats = ImageUtilities.ConvertBytesToFloat(bytes);

                            Array.Copy(floats, 0, data, copyIndexStart, sampleSize);

                            copyIndexStart += sampleSize;
                        }
                        currentIndex++;
                    }

                    return new DataBatch<float>(data, sampleShape, batchSampleCount);
                }

                var totalSampleCount = imageFilePaths.Length;
                return new DataLoader<float>(LoadImageData, totalSampleCount);
            }
        }
    }
}
