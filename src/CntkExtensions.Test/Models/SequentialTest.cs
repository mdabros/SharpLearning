using System;
using CntkExtensions.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkExtensions.Test.Models
{
    [TestClass]
    public class SequentialTest
    {
        [TestMethod]
        public void Sequential_Use_Case()
        {
            var network = new Sequential();
            network.Add(x => Layers.Dense(x, units: 512));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: 10));
            network.Add(x => Layers.Softmax(x));

            network.Compile(p => Learners.MomentumSGD(p), 
               (t, p) => Losses.MeanSquaredError(t, p), 
               (t, p) => Metrics.Accuracy(t, p));
        }
    }
}
