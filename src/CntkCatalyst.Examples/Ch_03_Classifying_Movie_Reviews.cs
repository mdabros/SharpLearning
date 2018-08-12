using System;
using CntkExtensions;
using CntkExtensions.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples
{
    [TestClass]
    public class Ch_03_Classifying_Movie_Reviews
    {
        [TestMethod]
        public void Run()
        {
            // download and load+vectorize imdb dataset.

            var inputShape = new int[] { 1000 };
            var numberOfClasses = 1;

            var network = new Sequential(Layers.Input(inputShape));

            network.Add(x => Layers.Dense(x, units: 16));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: 16));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: numberOfClasses));
            network.Add(x => Layers.Sigmoid(x));

            network.Compile(p => Learners.RMSProp(p, learningRate: 0.001),
               (t, p) => Losses.BinaryCrossEntropy(t, p),
               (t, p) => Metrics.Accuracy(t, p));

            // Split data and fit model with validtion.
            
        }
    }
}
