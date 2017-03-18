using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.Providers;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class BatchNormalizationLayer : ILayerNew
    {
        readonly IBatchNormalization m_batchNorm;

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
        /// <param name="batchNorm"></param>
        public BatchNormalizationLayer(IBatchNormalization batchNorm)
        {
            if (batchNorm == null) { throw new ArgumentNullException(nameof(batchNorm)); }
            m_batchNorm = batchNorm;
        }
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        public void Forward(Executor executor)
        {
            var input = executor.GetTensor(Input);
            var output = executor.GetTensor(Output);

            var scale = executor.GetTensor(Scale);
            var bias = executor.GetTensor(Bias);

            var batchColumnMeans = executor.GetTensor(BatchColumnMeans);
            var batchcolumnVars = executor.GetTensor(BatchcolumnVars);
            var movingAverageMeans = executor.GetTensor(MovingAverageMeans);
            var movingAverageVariance = executor.GetTensor(MovingAverageVariance);

            bool isTraining = true; // needs to be set

            m_batchNorm.Forward(input, scale, bias,
                batchColumnMeans.Data, batchcolumnVars.Data,
                movingAverageMeans.Data, movingAverageVariance.Data,
                output, isTraining);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        public void Backward(Executor executor)
        {
            var input = executor.GetTensor(Input);
            var inputGradient = executor.GetGradient(Input);

            var scale = executor.GetTensor(Scale);
            var bias = executor.GetTensor(Bias);

            var scaleGradient = executor.GetGradient(Scale);
            var biasGradient = executor.GetGradient(Bias);

            var batchColumnMeans = executor.GetTensor(BatchColumnMeans);
            var batchcolumnVars = executor.GetTensor(BatchcolumnVars);

            var outputGradient = executor.GetTensor(Output);

            m_batchNorm.Backward(input, scale, bias,
                scaleGradient, biasGradient,
                batchColumnMeans.Data, batchcolumnVars.Data,
                outputGradient, inputGradient);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="excecutor"></param>
        public void Initialize(Variable inputVariable, Executor excecutor)
        {
            Input = inputVariable;
            Output = new Variable(inputVariable.Dimensions.ToArray());
            
            var c = inputVariable.Dimensions[1];
            var fanOutAndIn = inputVariable.DimensionOffSets[0]; // product of all dimensions except batch size.

            Scale = Variable.CreateTrainable(fanOutAndIn);
            excecutor.AssignTensor(Scale, () => 1.0f);
                
            Bias = Variable.CreateTrainable(fanOutAndIn);
  
            BatchColumnMeans = Variable.Create(c);
            BatchcolumnVars = Variable.Create(c);

            MovingAverageMeans = Variable.Create(c);
            MovingAverageVariance = Variable.Create(c);
            excecutor.AssignTensor(MovingAverageVariance, () => 1.0f);
        }
    }
}
