using System.Collections.Generic;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.InputOutput.DataSources;

namespace SharpLearning.InputOutput.Test.DataSources
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
            var observationsLoader = DataLoaders.Csv.FromText(
                DataSetUtilities.AptitudeData,
                s => CsvRowExtensions.DefaultF64Converter(s),
                columnNames: new[] { "AptitudeTestScore", "PreviousExperience_month" },
                sampleShape: new[] { 2 });

            // targets loader.
            var targetsLoader = DataLoaders.Csv.FromText(
                DataSetUtilities.AptitudeData,
                s => CsvRowExtensions.DefaultF64Converter(s),
                columnNames: new[] { "Pass" },
                sampleShape: new[] { 1 });

            var idToLoader = new Dictionary<string, DataLoader<double>>
            {
                { "observations", observationsLoader },
                { "targets", targetsLoader },
            };

            return idToLoader;
        }

        static Dictionary<string, DataLoader<float>> CreateImageDataLoaders()
        {
            
            // targets loader.
            var targetsLoader = DataLoaders.Csv.FromParser(
                new CsvParser(() => new StreamReader(@"E:\DataSets\CIFAR10\test_map.csv"), separator: '\t'),
                s => float.Parse(s),
                columnNames: new[] { "target" },
                sampleShape: new[] { 1 });

            // images loader.
            var imageLoader = DataLoaders.Image.FromCsvParser(
                new CsvParser(() => new StreamReader(@"E:\DataSets\CIFAR10\test_map.csv"), separator: '\t'),
                imageNameColumnName: "filepath",
                imageDirectoryPath: @"E:\DataSets\CIFAR10\test",
                sampleShape: new int[] { 32, 32, 3 });

            var idToLoader = new Dictionary<string, DataLoader<float>>
            {
                { "images", imageLoader },
                { "targets", targetsLoader },
            };

            return idToLoader;
        }
    }
}
