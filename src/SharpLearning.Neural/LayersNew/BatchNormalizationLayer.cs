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
        public void Forward(Executor executor)
        {
            bool isTraining = true; // needs to be set

            BatchNormalization.Forward(Input, Scale, Bias,
                BatchColumnMeans, BatchcolumnVars,
                MovingAverageMeans, MovingAverageVariance,
                executor, isTraining, Output);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        public void Backward(Executor executor)
        {
            BatchNormalization.Backward(Input, Scale, Bias,
                BatchColumnMeans, BatchcolumnVars,
                executor, Output);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="excecutor"></param>
        /// <param name="random"></param>
        /// <param name="initializtion"></param>
        public void Initialize(Variable inputVariable, Executor excecutor, Random random, 
            Initialization initializtion = Initialization.GlorotUniform)
        {
            Input = inputVariable;
            Output = new Variable(inputVariable.Dimensions.ToArray());
            
            var c = inputVariable.Dimensions[1];
            var fanOutAndIn = inputVariable.DimensionOffSets[0]; // product of all dimensions except batch size.

            Scale = Variable.CreateTrainable(fanOutAndIn);
            excecutor.AssignTensor(Scale, () => 1.0);
                
            Bias = Variable.CreateTrainable(fanOutAndIn);
            excecutor.AssignTensor(Bias, () => 0.0);

            BatchColumnMeans = Variable.Create(c);
            BatchcolumnVars = Variable.Create(c);

            MovingAverageMeans = Variable.Create(c);
            MovingAverageVariance = Variable.Create(c);
            excecutor.AssignTensor(MovingAverageVariance, () => 1.0);
        }
    }
}
