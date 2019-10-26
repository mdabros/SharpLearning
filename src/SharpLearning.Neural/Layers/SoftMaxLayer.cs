using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// SoftMax Layer.
    /// The Softmax classifier is the generalization of the binary logistic regression classifier to multiple classes. 
    /// Unlike the SVM which treats the outputs as (uncalibrated and possibly difficult to interpret) scores for each class, 
    /// the Softmax classifier gives a slightly more intuitive output (normalized class probabilities.
    /// However, the softmax might sacrifice accuracy in order to achieve better probabilities.
    /// </summary>
    [Serializable]
    public sealed class SoftMaxLayer
        : ILayer
        , IOutputLayer
        , IClassificationLayer
    {
        Matrix<float> OutputActivations;
        Matrix<float> m_delta;

        /// <summary>
        /// 
        /// </summary>
        public int NumberOfClasses;

        /// <summary>
        /// 
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public int Height { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public int Depth { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Activation ActivationFunc { get; set; }

        /// <summary>
        /// The Softmax classifier is the generalization of the binary logistic regression classifier to multiple classes. 
        /// Unlike the SVM which treats the outputs as (uncalibrated and possibly difficult to interpret) scores for each class, 
        /// the Softmax classifier gives a slightly more intuitive output (normalized class probabilities.
        /// However, the softmax might sacrifice accuracy in order to achieve better propabilities.
        /// </summary>
        /// <param name="numberOfClasses"></param>
        public SoftMaxLayer(int numberOfClasses)
        {
            if (numberOfClasses < 2) { throw new ArgumentException("numberOfClasses is less than 2: " + numberOfClasses); }

            ActivationFunc = Activation.Undefined;
            NumberOfClasses = numberOfClasses;
            Height = 1;
            Depth = numberOfClasses;
            Width = 1;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="delta"></param>
        public Matrix<float> Backward(Matrix<float> delta)
        {
            delta.Subtract(OutputActivations, m_delta);
            m_delta.Multiply(-1f, m_delta);

            return m_delta;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix<float> Forward(Matrix<float> input)
        {
            input.CopyTo(OutputActivations);
            SoftMax(OutputActivations);

            return OutputActivations;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="inputDepth"></param>
        /// <param name="batchSize"></param>
        /// <param name="initializtion"></param>
        /// <param name="random"></param>

        public void Initialize(int inputWidth, int inputHeight, int inputDepth, int batchSize, Initialization initializtion, Random random)
        {
            OutputActivations = Matrix<float>.Build.Dense(batchSize, NumberOfClasses);
            m_delta = Matrix<float>.Build.Dense(batchSize, NumberOfClasses);
        }

        /// <summary>
        /// Softmax activation for neural net.
        /// </summary>
        /// <param name="x"></param>
        public void SoftMax(Matrix<float> x)
        {
            var xData = x.Data();
            var rows = x.RowCount;
            var cols = x.ColumnCount;

            for (int row = 0; row < x.RowCount; row++)
            {
                var rowSum = 0.0f;
                var max = double.MinValue;

                for (int col = 0; col < x.ColumnCount; ++col)
                {
                    var index = col * rows + row;
                    var value = xData[index];
                    if (value > max)
                    {
                        max = value;
                    }
                }

                for (int col = 0; col < x.ColumnCount; ++col)
                {
                    var index = col * rows + row;

                    var value = (float)Math.Exp(xData[index] - max);
                    rowSum += value;
                    xData[index] = value;
                }

                for (int col = 0; col < x.ColumnCount; ++col)
                {
                    var index = col * rows + row;
                    xData[index] = xData[index] / rowSum;
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            // classification layer does not have any parameters or gradients.
        }

        /// <summary>
        /// Copies a minimal version of the layer to be used in a model for predictions.
        /// </summary>
        /// <param name="layers"></param>
        public void CopyLayerForPredictionModel(List<ILayer> layers)
        {
            var batchSize = 1;
            var copy = new SoftMaxLayer(NumberOfClasses);
            copy.OutputActivations = Matrix<float>.Build.Dense(batchSize, NumberOfClasses);

            layers.Add(copy);
        }
    }
}
