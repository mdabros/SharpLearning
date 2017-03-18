using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Neural.LayersNew;
using SharpLearning.Containers.Tensors;

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

        public static void CheckLayer(ILayerNew layer, Executor executor, Variable inputShape, float epsilon, Random random)
        {
            var accuracyCondition = 1e-2;

            var input = executor.GetTensor(inputShape);
            input.Map(v => (float)random.NextDouble());

            layer.Forward(executor);
            layer.Backward(executor);

            var parameters = executor.GetTensor(inputShape).Data;
            var gradients = executor.GetGradient(inputShape).Data;

            var output1 = new float[parameters.Length];
            var output2 = new float[parameters.Length];

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
                var gradient = output1.Zip(output2, 
                    (v1, v2) => (v1 - v2) / (2.0f * epsilon)).Sum();

                var actual = gradients[i];
                Assert.AreEqual(gradient, actual, accuracyCondition);
            }
        }
    }
}
