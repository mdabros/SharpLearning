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

        public static void CheckLayer(ILayerNew layer, Executor executor, Variable inputVariable, float epsilon, Random random)
        {
            var accuracyCondition = 1e-2;

            var fans = new FanInFanOut(layer.Input.DimensionOffSets[0],
                layer.Output.DimensionOffSets[0]);

            // set input to 1
            var input = executor.GetTensor(inputVariable);
            input.Map(v => (float)random.NextDouble());

            layer.Forward(executor);

            // set output gradients to 1
            executor.GetGradient(layer.Output).Map(v => 1.0f); 

            layer.Backward(executor);

            var parameters = executor.GetTensor(inputVariable).Data;
            var gradients = executor.GetGradient(inputVariable).Data;

            var output1 = new float[layer.Output.ElementCount];
            var output2 = new float[layer.Output.ElementCount];

            for (int i = 0; i < parameters.Length; i++)
            {
                var oldValue = parameters[i];

                parameters[i] = oldValue + epsilon;
                layer.Forward(executor);
                executor.GetTensor(layer.Output).Data.CopyTo(output1, 0);

                parameters[i] = oldValue - epsilon;
                layer.Forward(executor);
                executor.GetTensor(layer.Output).Data.CopyTo(output2, 0);

                parameters[i] = oldValue;

                // approximated gradient
                var diff = output1.Zip(output2,
                    (v1, v2) => (v1 - v2)).ToArray();

                var gradient = diff.Select(v => v / (2.0f * epsilon)).Sum();

                var actual = gradients[i];
                Assert.AreEqual(gradient, actual, accuracyCondition);
            }
        }

        public static void CheckLayerParameters(ILayerNew layer, Executor executor, Variable inputVariable, float epsilon, Random random)
        {
            var accuracyCondition = 1e-2;

            var fans = new FanInFanOut(layer.Input.DimensionOffSets[0],
                layer.Output.DimensionOffSets[0]);

            // set input to 1
            var input = executor.GetTensor(inputVariable);
            input.Map(v => 1.0f);

            layer.Forward(executor);

            // set output gradients to 1
            executor.GetGradient(layer.Output).Map(v => 1.0f);

            layer.Backward(executor);

            var output1 = new float[layer.Output.ElementCount];
            var output2 = new float[layer.Output.ElementCount];

            var trainableParameters = new List<Data>();
            executor.GetTrainableParameters(trainableParameters);

            foreach (var data in trainableParameters)
            {
                var parameters = data.Tensor.Data;
                var gradients = data.Gradient.Data;

                for (int i = 0; i < parameters.Length; i++)
                {
                    var oldValue = parameters[i];

                    parameters[i] = oldValue + epsilon;
                    layer.Forward(executor);
                    executor.GetTensor(layer.Output).Data.CopyTo(output1, 0);

                    parameters[i] = oldValue - epsilon;
                    layer.Forward(executor);
                    executor.GetTensor(layer.Output).Data.CopyTo(output2, 0);

                    parameters[i] = oldValue;

                    // approximated gradient
                    var diff = output1.Zip(output2,
                        (v1, v2) => (v1 - v2)).ToArray();

                    var gradient = diff.Select(v => v / (2.0f * epsilon)).Sum();

                    var actual = gradients[i];
                    Assert.AreEqual(gradient, actual, accuracyCondition);
                }
            }
        }
    }
}
