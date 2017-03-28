using System;
using System.Linq;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Providers.DotNetOp;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class BatchNormalizationLayer : ILayerNew
    {
        /// <summary>
        /// Layer output shape
        /// </summary>
        public Variable Output { get; set; }

        /// <summary>
        /// Layer Input shape
        /// </summary>
        public Variable Input { get; set; }

        Variable BatchColumnMeans;
        Variable BatchcolumnVars;

        Variable MovingAverageMeans;
        Variable MovingAverageVariance;

        Variable Scale;
        Variable Bias;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        /// <param name="training"></param>
        public void Forward(NeuralNetStorage executor, bool training = true)
        {
            BatchNormalization.Forward(Input, Scale, Bias,
                BatchColumnMeans, BatchcolumnVars,
                MovingAverageMeans, MovingAverageVariance,
                executor, training, Output);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        public void Backward(NeuralNetStorage executor)
        {
            BatchNormalization.Backward(Input, Scale, Bias,
                BatchColumnMeans, BatchcolumnVars,
                executor, Output);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="storage"></param>
        /// <param name="random"></param>
        /// <param name="initializtion"></param>
        public void Initialize(Variable inputVariable, NeuralNetStorage storage, Random random, 
            Initialization initializtion = Initialization.GlorotUniform)
        {
            UpdateDimensions(inputVariable);

            var c = inputVariable.Dimensions[1];
            var fanOutAndIn = inputVariable.DimensionOffSets[0]; // product of all dimensions except batch size.

            Scale = Variable.CreateTrainable(fanOutAndIn);
            storage.AssignTensor(Scale, () => 1.0);
                
            Bias = Variable.CreateTrainable(fanOutAndIn);
            storage.AssignTensor(Bias, () => 0.0);

            BatchColumnMeans = Variable.Create(c);
            BatchcolumnVars = Variable.Create(c);

            MovingAverageMeans = Variable.CreatePreservable(c);
            MovingAverageVariance = Variable.CreatePreservable(c);
            storage.AssignTensor(MovingAverageVariance, () => 1.0);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(Variable input)
        {
            Input = input;
            Output = Variable.Create(input.Dimensions.ToArray());
        }
    }
}
