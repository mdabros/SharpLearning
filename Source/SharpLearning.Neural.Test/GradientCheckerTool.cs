using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Layers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Neural.Test
{
    public static class GradientCheckTools
    {
        public static void CheckLayer(ILayer layer, int fanInWidth, int fanInHeight, int fanInDepth, int batchSize, float epsilon, Random random)
        {
            var accuracyCondition = 1e-2;
            layer.Initialize(fanInWidth, fanInHeight, fanInDepth, batchSize, random);

            var fanIn = fanInWidth * fanInHeight * fanInDepth;
            var fanOut = layer.Width * layer.Height * layer.Depth;

            // Forward pass - set input activation in layer
            var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
            layer.Forward(input);

            // Set delta to 1
            var delta = Matrix<float>.Build.Dense(batchSize, fanOut, 1.0f);

            // Backward pass to calculate gradients
            layer.Backward(delta);

            // get weights and gradients
            var weightGradients = layer.GetGradients().Weights;
            var weightParameters = layer.GetParameters().Weights;
     
            var output1 = Matrix<float>.Build.Dense(batchSize, fanOut);
            var output2 = Matrix<float>.Build.Dense(batchSize, fanOut);

            // Check weights
            for (var y = 0; y < weightParameters.ColumnCount; y++)
            {
                for (var x = 0; x < weightParameters.RowCount; x++)
                {
                    output1.Clear();
                    output2.Clear();

                    var oldValue = weightParameters[x, y];

                    weightParameters[x, y] = oldValue + epsilon;
                    layer.Forward(input).CopyTo(output1);
                    weightParameters[x, y] = oldValue - epsilon;
                    layer.Forward(input).CopyTo(output2);

                    weightParameters[x, y] = oldValue;

                    output1.Subtract(output2, output1); // output1 = output1 - output2

                    var grad = output1.ToRowMajorArray().Select(f => f / (2.0f * epsilon));
                    var gradient = grad.Sum(); // approximated gradient
                    var actual = weightGradients[x, y];

                    Assert.AreEqual(gradient, actual, accuracyCondition);
                }
            }

            // Check biases
            var biasesGradients = layer.GetGradients().Bias;
            var biasesParameters = layer.GetParameters().Bias;

            if (biasesGradients != null)
            {
                for (int i = 0; i < biasesGradients.Count; i++)
                {
                    output1.Clear();
                    output2.Clear();

                    var oldValue = biasesParameters[i];

                    biasesParameters[i] = oldValue + epsilon;
                    layer.Forward(input).CopyTo(output1);
                    biasesParameters[i] = oldValue - epsilon;
                    layer.Forward(input).CopyTo(output2);

                    biasesParameters[i] = oldValue;

                    output1.Subtract(output2, output1); // output1 = output1 - output2

                    var grad = output1.ToRowMajorArray().Select(f => f / (2.0f * epsilon));

                    var gradient = grad.Sum();
                    var actual = biasesGradients[i];

                    Assert.AreEqual(gradient, actual, accuracyCondition);
                }
            }
        }

        static Matrix<float> GetWeights(ILayer layer)
        {
            var parametersAndGradients = new List<ParametersAndGradients>();
            layer.AddParameresAndGradients(parametersAndGradients);
            return parametersAndGradients.Single().Parameters.Weights;
        }
    }
}
