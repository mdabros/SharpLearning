using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// SvmLayer.
    /// Because the SVM is a margin classifier, it is happy once the margins are satisfied 
    /// and it does not micromanage the exact scores beyond this constraint.
    /// This can be an advantage when the overall goal is the best possible accuracy. And probability estimates is less important.
    /// </summary>
    [Serializable]
    public sealed class SvmLayer 
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
        /// Because the SVM is a margin classifier, it is happy once the margins are satisfied 
        /// and it does not micromanage the exact scores beyond this constraint.
        /// This can be an advantage when the overall goal is the best possible accuracy. And probability estimates is less important.
        /// </summary>
        /// <param name="numberOfClasses"></param>
        public SvmLayer(int numberOfClasses)
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
            const double margin = 1.0;
            var batchSize = delta.RowCount;

            m_delta.Clear();
            for (int batchItem = 0; batchItem < batchSize; batchItem++) // for each batch item
            {
                var maxTarget = 0.0;
                var maxTargetIndex = 0;
                for (int col = 0; col < delta.ColumnCount; col++)
                {
                    var targetValue = delta.At(batchItem, col);
                    if (targetValue > maxTarget)
                    {
                        maxTarget = targetValue;
                        maxTargetIndex = col;
                    }
                }

                var maxTargetScore = OutputActivations.At(batchItem, maxTargetIndex);
                for (int i = 0; i < OutputActivations.ColumnCount; i++)
                {
                    if(i == maxTargetIndex) { continue; }

                    // The score of the target should be higher than he score of any other class, by a margin
                    var diff = -maxTargetScore + OutputActivations.At(batchItem, i) + margin;
                    if(diff > 0)
                    {
                        m_delta[batchItem, i] += 1;
                        m_delta[batchItem, maxTargetIndex] -= 1;
                    }
                }
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
            OutputActivations = Matrix<float>.Build.Dense(batchSize, NumberOfClasses);
            m_delta = Matrix<float>.Build.Dense(batchSize, NumberOfClasses);
        }

        /// <summary>
        /// SvmLayer layer does not have any parameters or gradients.
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            // SvmLayer layer does not have any parameters or gradients.
        }

        /// <summary>
        /// Copies a minimal version of the layer to be used in a model for predictions.
        /// </summary>
        /// <param name="layers"></param>
        public void CopyLayerForPredictionModel(List<ILayer> layers)
        {
            var batchSize = 1;
            var copy = new SvmLayer(NumberOfClasses);
            copy.OutputActivations = Matrix<float>.Build.Dense(batchSize, NumberOfClasses);

            layers.Add(copy);
        }
    }
}
