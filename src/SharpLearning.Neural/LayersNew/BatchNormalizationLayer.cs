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
        public TensorShape Output { get; set; }

        /// <summary>
        /// Layer Input shape
        /// </summary>
        public TensorShape Input { get; set; }

        TensorShape BatchColumnMeans;
        TensorShape BatchcolumnVars;

        TensorShape MovingAverageMeans;
        TensorShape MovingAverageVariance;

        TensorShape Scale;
        TensorShape Bias;

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
        /// <param name="inputShape"></param>
        /// <param name="excecutor"></param>
        public void Initialize(TensorShape inputShape, Executor excecutor)
        {
            Input = inputShape;
            Output = new TensorShape(inputShape.Dimensions.ToArray());
            
            var c = inputShape.Dimensions[1];
            var fanOutAndIn = inputShape.DimensionOffSets[0]; // product of all dimensions except batch size.

            Scale = new TensorShape(fanOutAndIn);
            excecutor.GetTensor(Scale)
                .Map(v => 1.0f);
            excecutor.GetGradient(Scale) // should gradients also be mapped to 1?
                .Map(v => 1.0f);

            Bias = new TensorShape(fanOutAndIn);
            excecutor.GetTensor(Scale)
                .Map(v => 0.0f);

            BatchColumnMeans = new TensorShape(c);
            BatchcolumnVars = new TensorShape(c);

            MovingAverageMeans = new TensorShape(c);
            MovingAverageVariance = new TensorShape(c);
            excecutor.GetTensor(MovingAverageVariance)
                .Map(v => 1.0f);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameters"></param>
        public void GetTrainableParameterShapes(List<TensorShape> parameters)
        {
            parameters.Add(Scale);
            parameters.Add(Bias);
        }
    }
}
