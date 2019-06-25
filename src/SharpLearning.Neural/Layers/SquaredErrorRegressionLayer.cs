using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// Regression layer using the squared error as loss function.
    /// </summary>
    [Serializable]
    public sealed class SquaredErrorRegressionLayer : ILayer, IOutputLayer, IRegressionLayer
    {
        Matrix<float> OutputActivations;
        Matrix<float> m_delta;

        /// <summary>
        /// 
        /// </summary>
        public int NumberOfTargets;

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
        /// Regression layer using the squared error as loss function.
        /// </summary>
        /// <param name="numberOfTargets">If more than one regression target. Default is 1</param>
        public SquaredErrorRegressionLayer(int numberOfTargets = 1)
        {
            if (numberOfTargets < 1) { throw new ArgumentException("numberOfClasses is less than 1: " + numberOfTargets); }

            ActivationFunc = Activation.Undefined;
            NumberOfTargets = numberOfTargets;
            Height = 1;
            Depth = numberOfTargets;
            Width = 1;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="delta"></param>
        /// <returns></returns>
        public Matrix<float> Backward(Matrix<float> delta)
        {
            var targetsArray = delta.Data();
            var predictionsArray = OutputActivations.Data();
            var deltaData = m_delta.Data();

            for (int i = 0; i < targetsArray.Length; i++)
            {
                deltaData[i] = (predictionsArray[i] - targetsArray[i]);
            }

            return m_delta;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix<float> Forward(Matrix<float> input)
        {
            input.CopyTo(OutputActivations); // do nothing, output raw scores
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

        public void Initialize(int inputWidth, int inputHeight, int inputDepth, int batchSize, 
            Initialization initializtion, Random random)
        {
            OutputActivations = Matrix<float>.Build.Dense(batchSize, NumberOfTargets);
            m_delta = Matrix<float>.Build.Dense(batchSize, NumberOfTargets);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="layers"></param>
        public void CopyLayerForPredictionModel(List<ILayer> layers)
        {
            var batchSize = 1;
            var copy = new SquaredErrorRegressionLayer(NumberOfTargets);
            copy.OutputActivations = Matrix<float>.Build.Dense(batchSize, NumberOfTargets);

            layers.Add(copy);
        }

        /// <summary>
        /// SquaredErrorRegressionLayer layer does not have any parameters or gradients.
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            // SquaredErrorRegressionLayer layer does not have any parameters or gradients.
        }
    }
}
