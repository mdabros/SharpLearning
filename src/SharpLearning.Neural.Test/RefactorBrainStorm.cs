using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Neural.Test
{
    [TestClass]
    public class RefactorBrainStormTest
    {
        interface ILayer
        {
            string Name { get; }
            void Forward();
            void Backward();
        }

        class DenseLayer : ILayer
        {
            public string Name { get; }

            public void Forward()
            {
            }

            public void Backward()
            {
            }
        }

        interface ILoss
        {
            DenseTensor<double> sampleLosses(DenseTensor<double> targets, DenseTensor<double> predictions);
            double AccumulateSampleLoss(DenseTensor<double> sampleLosses);
        }

        class MeanSquareLoss : ILoss
        {
            public DenseTensor<double> sampleLosses(DenseTensor<double> targets, DenseTensor<double> predictions)
            {
                CheckDimensions(targets, predictions);

                var losses = new DenseTensor<double>(targets.Dimensions);

                for (int i = 0; i < targets.Length; i++)
                {
                    losses[i] = predictions[i] - targets[i];
                }

                return losses;
            }

            public double AccumulateSampleLoss(DenseTensor<double> sampleLosses)
            {
                var accumulatedLoss = 0.0;

                for (int i = 0; i < sampleLosses.Length; i++)
                {
                    var sampleLoss = sampleLosses[i];
                    accumulatedLoss += sampleLoss * sampleLoss;
                }

                accumulatedLoss = accumulatedLoss / sampleLosses.Length;

                return accumulatedLoss;
            }

            static void CheckDimensions(Tensor<double> targets, Tensor<double> predictions)
            {
                if (targets.Dimensions != predictions.Dimensions)
                {
                    throw new ArgumentException($"Target dimensions: {targets.Dimensions.ToArray()} differs from "
                        + $" Prediction dimensions: {predictions.Dimensions.ToArray()}");
                }
            }
        }

        class NeuralNet
        {
            public NeuralNet(List<ILayer> layers)
            {
                Layers = layers ?? throw new ArgumentNullException(nameof(layers));
            }

            public List<ILayer> Layers { get; }

            public void Forward()
            {
                Layers.ForEach(l => l.Forward());
            }

            public void Backward()
            {
                Layers.ForEach(l => l.Backward());
            }
        }

        class Optimizer
        {

        }

        [TestMethod]
        public void RefactorBrainStorm()
        {

        }
    }
}
