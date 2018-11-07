using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CNTK;
using CntkCatalyst.LayerFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples
{
    /// <summary>
    /// Example from Chapter 3.5: Classifying movie reviews:
    /// https://github.com/mdabros/deep-learning-with-python-notebooks/blob/master/3.5-classifying-movie-reviews.ipynb
    ///
    ///
    /// This example needs manual download of the IMDB dataset in sparse CNTK format.
    /// The data can found following the link below, 
    /// and then downloading from the "Get the Code" link at the top of the page:
    /// https://msdn.microsoft.com/en-us/magazine/mt830362
    /// </summary>
    [TestClass]
    public class Ch_03_Classifying_Movie_Reviews
    {
        [TestMethod]
        public void Run()
        {
            // Prepare data
            var baseDataDirectoryPath = @"E:\DataSets\Imdb";
            var trainFilePath = Path.Combine(baseDataDirectoryPath, "imdb_sparse_train_50w.txt");
            var testFilePath = Path.Combine(baseDataDirectoryPath, "imdb_sparse_test_50w.txt");

            // Define the input and output shape.
            var inputShape = new int[] { 129888 + 4 }; // Number distinct input words + offset for one-hot, sparse
            var numberOfClasses = 2;
            var outputShape = new int[] { numberOfClasses };

            // Setup minibatch sources.
            // Network will be trained using the training set,
            // and tested using the test set.
            var featuresName = "x";
            var targetsName = "y";

            // The order of the training data is randomize.
            var train = CreateMinibatchSource(trainFilePath, featuresName, targetsName,
                numberOfClasses, inputShape, randomize: true);
            var trainingSource = new CntkMinibatchSource(train, featuresName, targetsName);

            // Notice randomization is switched off for test data.
            var test = CreateMinibatchSource(testFilePath, featuresName, targetsName,
                numberOfClasses, inputShape, randomize: false);
            var testSource = new CntkMinibatchSource(test, featuresName, targetsName);

            // Define data type and device for the model.
            var dataType = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            // Create the architecture.
            //var input = CNTKLib.SequenceLast(Layers.Input(inputShape, dataType));
            var network = Layers.Input(inputShape, dataType, isSparse: true)
                .Dense(16, device, dataType)
                .ReLU()
                .Dense(16, device, dataType)
                .ReLU()
                .Dense(numberOfClasses, device, dataType)
                .Softmax();

            // Since we are processing sequence data, 
            // wrap network in sequenceLast.
            network = CNTKLib.SequenceLast(network);

            var targetVariable = Variable.InputVariable(outputShape, dataType,
                dynamicAxes: new List<Axis>() { Axis.DefaultBatchAxis() },
                isSparse: false);

            // Create the network.
            var model = new Sequential(network, dataType, device);

            // Compile the network with the selected learner, loss and metric,
            // and custom target variable.
            model.Compile(p => Learners.Adam(p),
               (p, t) => Losses.CategoricalCrossEntropy(p, t),
               (p, t) => Metrics.Accuracy(p, t),
               targetVariable);

            // Train the model using the training set.
            var history = model.FitFromMinibatchSource(trainingSource, epochs: 20, batchSize: 512,
                validationMinibatchSource: testSource);

            // Trace loss and validation history
            TraceLossValidationHistory(history);

            // Evaluate the model using the test set.
            (var loss, var metric) = model.EvaluateFromMinibatchSource(testSource);

            // Write the test set loss and metric to debug output.
            Trace.WriteLine($"Test set - Loss: {loss}, Metric: {metric}");

            //// Write first ten predictions
            //var predictions = model.Predict(testObservations.GetSamples(Enumerable.Range(0, 10).ToArray()));

            //// Use tensor data directly, since only 1 element pr. sample.
            //Trace.WriteLine($"Predictions: [{string.Join(", ", predictions.Data)}]");          
        }

        MinibatchSource CreateMinibatchSource(string filePath, string featuresName, string targetsName,
            int numberOfClasses, int[] inputShape, bool randomize)
        {
            var inputSize = inputShape.Aggregate((d1, d2) => d1 * d2);
            var streamConfigurations = new StreamConfigurationVector
            {
                new StreamConfiguration(featuresName, inputSize, isSparse: true),
                new StreamConfiguration(targetsName, numberOfClasses, isSparse: false)
            };

            var deserializer = CNTKLib.CTFDeserializer(filePath, streamConfigurations);

            var minibatchSourceConfig = new MinibatchSourceConfig(new DictionaryVector() { deserializer });
            return CNTKLib.CreateCompositeMinibatchSource(minibatchSourceConfig);
        }

        static void TraceLossValidationHistory(Dictionary<string, List<float>> history)
        {
            foreach (var item in history)
            {
                Trace.WriteLine($"{item.Key}: [{string.Join(", ", item.Value)}]");
            }
        }
    }
}
