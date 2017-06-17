using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Test
{
    public static class GradientCheckTools
    {
        public static void CheckLayer(ILayer layer, int fanInWidth, int fanInHeight, int fanInDepth, int batchSize, float epsilon, Random random)
        {
            var accuracyCondition = 1e-2;
            layer.Initialize(fanInWidth, fanInHeight, fanInDepth, batchSize, Initialization.GlorotUniform, random);

            var fanIn = fanInWidth * fanInHeight * fanInDepth;
            var fanOut = layer.Width * layer.Height * layer.Depth;

            // Forward pass - set input activation in layer
            var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
            layer.Forward(input);

            // Set delta to 1
            var delta = Matrix<float>.Build.Dense(batchSize, fanOut, 1.0f);

            // Backward pass to calculate gradients
            var computedGradient = layer.Backward(delta);

            var gradients = computedGradient.Data();
            var parameters = input.Data();

            var output1 = Matrix<float>.Build.Dense(batchSize, fanOut);
            var output2 = Matrix<float>.Build.Dense(batchSize, fanOut);

            for (int i = 0; i < parameters.Length; i++)
            {
                output1.Clear();
                output2.Clear();

                var oldValue = parameters[i];

                parameters[i] = oldValue + epsilon;
                layer.Forward(input).CopyTo(output1);
                parameters[i] = oldValue - epsilon;
                layer.Forward(input).CopyTo(output2);

                parameters[i] = oldValue;

                output1.Subtract(output2, output1); // output1 = output1 - output2

                var grad = output1.ToRowMajorArray().Select(f => f / (2.0f * epsilon));
                var gradient = grad.Sum(); // approximated gradient
                var actual = gradients[i];

                Assert.AreEqual(gradient, actual, accuracyCondition);
            }
        }

        public static void CheckLayer(ILayerNew layer, NeuralNetStorage storage, Variable inputVariable, Random random)
        {
            var maxAllowedRelativeError = 1e-7;
            var epsilon = 1e-5;
            var batchSize = inputVariable.Dimensions[0];

            var input = storage.GetTensor(inputVariable);
            input.Map(v => random.NextDouble());

            layer.Forward(storage);

            // set output gradients to 1
            storage.GetGradient(layer.Output).Map(v => 1.0); 

            layer.Backward(storage);

            var parameters = storage.GetTensor(inputVariable).Data;
            var gradients = storage.GetGradient(inputVariable).Data;

            var output1 = new double[layer.Output.ElementCount];
            var output2 = new double[layer.Output.ElementCount];

            for (int i = 0; i < parameters.Length; i++)
            {
                var oldValue = parameters[i];

                parameters[i] = oldValue + epsilon;
                layer.Forward(storage);
                storage.GetTensor(layer.Output).Data.CopyTo(output1, 0);

                parameters[i] = oldValue - epsilon;
                layer.Forward(storage);
                storage.GetTensor(layer.Output).Data.CopyTo(output2, 0);

                parameters[i] = oldValue;

                // approximated gradient
                var diff = output1.Zip(output2,
                    (v1, v2) => (v1 - v2)).ToArray();

                var sum = diff.Sum();
                var distinct = diff.Distinct().ToArray();

                var approxGradient = diff.Select(v => v / (2.0 * epsilon)).Sum();
                var computedGradient = gradients[i];

                var error = RelativeError(approxGradient, computedGradient);

                Assert.IsTrue(error < maxAllowedRelativeError);
            }
        }

        public static void CheckLayerParameters(ILayerNew layer, NeuralNetStorage storage, Variable inputVariable, Random random)
        {
            var maxAllowedRelativeError = 1e-7;
            var epsilon = 1e-5;
            var batchSize = inputVariable.Dimensions[0];

            // set input to 1
            var input = storage.GetTensor(inputVariable);
            input.Map(v => 1.0);

            layer.Forward(storage);

            // set output gradients to 1
            storage.GetGradient(layer.Output).Map(v => 1.0);

            layer.Backward(storage);

            var output1 = new double[layer.Output.ElementCount];
            var output2 = new double[layer.Output.ElementCount];

            var trainableParameters = new List<Data<double>>();
            storage.GetTrainableParameters(trainableParameters);

            foreach (var data in trainableParameters)
            {
                var parameters = data.Tensor.Data;
                var gradients = data.Gradient.Data;

                for (int i = 0; i < parameters.Length; i++)
                {
                    var oldValue = parameters[i];

                    parameters[i] = oldValue + epsilon;
                    layer.Forward(storage);
                    storage.GetTensor(layer.Output).Data.CopyTo(output1, 0);

                    parameters[i] = oldValue - epsilon;
                    layer.Forward(storage);
                    storage.GetTensor(layer.Output).Data.CopyTo(output2, 0);

                    parameters[i] = oldValue;

                    // approximated gradient
                    var diff = output1.Zip(output2,
                        (v1, v2) => (v1 - v2)).ToArray();

                    var approxGradient = diff.Select(v => v / (2.0 * epsilon)).Sum();
                    var computedGradient = gradients[i];
                    
                    var error = RelativeError(approxGradient, computedGradient);

                    Assert.IsTrue(error < maxAllowedRelativeError);
                }
            }
        }

        static double RelativeError(double approximatedGradient, double computedGradient)
        {
            if (approximatedGradient == 0.0 && computedGradient == 0.0)
            {
                // in case both gradients are zero, skip the calculation.
                return 0.0;
            }

            return Math.Abs(approximatedGradient - computedGradient) / 
                Math.Max(Math.Abs(approximatedGradient), Math.Abs(computedGradient));
        }
    }
}
