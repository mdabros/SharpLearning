using System.Collections.Generic;
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
            var idToLoaders = CreateDataLoaders();

            var sut = new DataSource<double>(idToLoaders, batchSize: 3, 
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
            var idToLoaders = CreateDataLoaders();

            var sut = new DataSource<double>(idToLoaders, batchSize: 3,
                shuffle: true, seed: 32);

            var (actual, isSweepEnd) = sut.GetNextBatch();

            var expected = new Dictionary<string, DataBatch<double>>()
            {
                { "observations", new DataBatch<double>(new double[] { 1, 12, 4, 24, 1, 8 }, new int[] { 2 }, 3) },
                { "targets", new DataBatch<double>(new double[] { 0, 1, 0 }, new int[] { 1 }, 3) },
            };

            AssertUtilities.AssertAreEqual(expected, actual);
        }

        static Dictionary<string, DataLoader<double>> CreateDataLoaders()
        {
            // observations loader.
            var observationsLoader = DataLoaders.FromCsv(
                CsvParser.FromText(DataSetUtilities.AptitudeData),
                s => CsvRowExtensions.DefaultF64Converter(s),
                columnNames: new[] { "AptitudeTestScore", "PreviousExperience_month" },
                sampleShape: new[] { 2 });

            // targets loader.
            var targetsLoader = DataLoaders.FromCsv(
                CsvParser.FromText(DataSetUtilities.AptitudeData),
                s => CsvRowExtensions.DefaultF64Converter(s),
                columnNames: new[] { "Pass" },
                sampleShape: new[] { 1 });

            var idToLoaders = new Dictionary<string, DataLoader<double>>
            {
                { "observations", observationsLoader },
                { "targets", targetsLoader },
            };

            return idToLoaders;
        }
    }
}
