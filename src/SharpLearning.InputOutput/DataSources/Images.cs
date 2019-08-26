using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpLearning.InputOutput.Csv;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpLearning.InputOutput.DataSources
{
    public static class Images
    {
        /// <summary>
        /// Load images from csv image name columns, and image directory path.
        /// </summary>
        /// <param name="csvText"></param>
        /// <param name="imageNameColumnName"></param>
        /// <param name="imagesDirectoryPath"></param>
        /// <returns></returns>
        public static IEnumerable<ImageGetter<TPixel>> FromCsvText<TPixel>(string csvText,
            string imageNameColumnName,
            string imagesDirectoryPath) where TPixel : struct, IPixel<TPixel>
        {
            var parser = CsvParser.FromText(csvText);
            return FromCsvParser<TPixel>(parser, imageNameColumnName, imagesDirectoryPath);
        }

        /// <summary>
        /// Load images from csv image name columns, and image directory path.
        /// </summary>
        /// <param name="csvFilePath"></param>
        /// <param name="imageNameColumnName"></param>
        /// <param name="imagesDirectoryPath"></param>
        /// <returns></returns>
        public static IEnumerable<ImageGetter<TPixel>> FromCsvFile<TPixel>(string csvFilePath,
            string imageNameColumnName,
            string imagesDirectoryPath) where TPixel : struct, IPixel<TPixel>
        {
            var parser = CsvParser.FromFile(csvFilePath);
            return FromCsvParser<TPixel>(parser, imageNameColumnName, imagesDirectoryPath);
        }

        /// <summary>
        /// Load images from csv image name columns, and image directory path.
        /// </summary>
        /// <param name="parser"></param>
        /// <param name="imageNameColumnName"></param>
        /// <param name="imagesDirectoryPath"></param>
        /// <returns></returns>
        public static IEnumerable<ImageGetter<TPixel>> FromCsvParser<TPixel>(CsvParser parser,
            string imageNameColumnName,
            string imagesDirectoryPath) where TPixel : struct, IPixel<TPixel>
        {
            var imageFilePaths = parser.EnumerateRows(imageNameColumnName)
                .ToStringVector().Select(filename => Path.Combine(imagesDirectoryPath, filename))
                .ToArray();

            return ImageUtilities.EnumerateImages<TPixel>(imageFilePaths);
        }
    }
}
