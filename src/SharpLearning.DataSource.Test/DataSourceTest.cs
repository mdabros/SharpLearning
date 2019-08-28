using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpLearning.DataSource.Test.DataSources
{
    [TestClass]
    public class DataSourceTest
    {
        [TestMethod]
        public void DataSource_Csv_GetNextBatch()
        {
            var idToLoader = CreateCsvDataLoaders();

            var sut = new DataSource<double>(idToLoader, batchSize: 3, 
                shuffle: false, seed: 32);

            var (actual, isSweepEnd) = sut.GetNextBatch();
            var expected = new Dictionary<string, DataBatch<double>>()
            {
                { "observations", new DataBatch<double>(new double[] { 5, 6, 1, 15, 1, 12 }, new int[] { 2 }, 3) },
                { "targets", new DataBatch<double>(new double[] { 0, 0, 0 }, new int[] { 1 }, 3) },
            };

            AssertUtilities.AssertAreEqual(expected, actual);
        }

        [TestMethod]
        public void DataSource_Csv_GetNextBatch_Shuffle()
        {
            var idToLoader = CreateCsvDataLoaders();

            var sut = new DataSource<double>(idToLoader, batchSize: 3,
                shuffle: true, seed: 32);

            var (actual, isSweepEnd) = sut.GetNextBatch();

            var expected = new Dictionary<string, DataBatch<double>>()
            {
                { "observations", new DataBatch<double>(new double[] { 1, 12, 4, 24, 1, 8 }, new int[] { 2 }, 3) },
                { "targets", new DataBatch<double>(new double[] { 0, 1, 0 }, new int[] { 1 }, 3) },
            };

            AssertUtilities.AssertAreEqual(expected, actual);
        }

        [TestMethod]
        public void DataSource_Image_GetNextBatch()
        {
            // TODO: Add in-memory test 
            var idToLoader = CreateImageDataLoaders();

            var sut = new DataSource<float>(idToLoader, batchSize: 32,
                shuffle: false, seed: 32);

            var (actual, isSweepEnd) = sut.GetNextBatch();
        }

        static Dictionary<string, DataLoader<double>> CreateCsvDataLoaders()
        {
            // observations loader.
            var observationsLoader = CsvParser.FromText(DataSetUtilities.AptitudeData)
                .EnumerateRows("AptitudeTestScore", "PreviousExperience_month").Select(v => v.Values)
                .ToCsvDataLoader(columnParser: s => CsvRowExtensions.DefaultF64Converter(s), sampleShape: 2);

            // targets loader.
            var targetsLoader = CsvParser.FromText(DataSetUtilities.AptitudeData)
                .EnumerateRows("Pass").Select(v => v.Values)
                .ToCsvDataLoader(columnParser: s => CsvRowExtensions.DefaultF64Converter(s), sampleShape: 1);

            var idToLoader = new Dictionary<string, DataLoader<double>>
            {
                { "observations", observationsLoader },
                { "targets", targetsLoader },
            };

            return idToLoader;
        }

        static Dictionary<string, DataLoader<float>> CreateImageDataLoaders()
        {
            // targets data loader.
            var targetsLoader = CsvParser.FromFile(@"E:\DataSets\CIFAR10\test_map.csv", separator: '\t')
                .EnumerateRows("target").Select(v => v.Values)
                .ToCsvDataLoader(columnParser: s => float.Parse(s), sampleShape: 1);

            // enumerate images.
            var imagesDirectoryPath = @"E:\DataSets\CIFAR10\test";
            var imageFilePaths = CsvParser.FromFile(@"E:\DataSets\CIFAR10\test_map.csv", separator: '\t')
                .EnumerateRows("filepath").ToStringVector()
                .Select(filename => Path.Combine(imagesDirectoryPath, filename));

            // images data loader.
            var random = new Random(Seed: 232);
            var imagesLoader = DataLoaders.EnumerateImages<Rgba32>(imageFilePaths)
                // Add augmentations
                .Select(i => i.Resize(20, 20).Rotate(maxDegrees: 20, random))
                .ToImageDataLoader(pixelConverter: Converters.ToFloat, sampleShape: new[] { 20, 20, 3 });

            var idToLoader = new Dictionary<string, DataLoader<float>>
            {
                { "images", imagesLoader },
                { "targets", targetsLoader },
            };

            return idToLoader;
        }
    }
}
