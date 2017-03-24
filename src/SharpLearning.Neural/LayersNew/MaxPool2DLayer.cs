using System;
using System.Linq;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Providers.DotNetOp;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class MaxPool2DLayer : ILayerNew
    {
        readonly MaxPool2DDescriptor m_descriptor;

        int[][] m_switchX;
        int[][] m_switchY;

        readonly BorderMode m_borderMode;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="poolH">Height of the pooling window</param>
        /// <param name="poolW">Width of the pooling window</param>
        /// <param name="strideH">Pooling vertical stride</param>
        /// <param name="strideW">Pooling horizontal stride</param>
        /// <param name="padH">Size of vertical padding</param>
        /// <param name="padW">Size of horizontal padding</param>
        public MaxPool2DLayer(int poolH, int poolW,
            int strideH, int strideW,
            int padH, int padW)
        {
            m_descriptor = new MaxPool2DDescriptor(poolH, poolW,
            strideH, strideW,
            padH, padW);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="poolH">Height of the pooling window</param>
        /// <param name="poolW">Width of the pooling window</param>
        /// <param name="strideH">Pooling vertical stride</param>
        /// <param name="strideW">Pooling horizontal stride</param>
        /// <param name="borderMode"></param>
        public MaxPool2DLayer(int poolH, int poolW, int strideH = 2, int strideW = 2, 
            BorderMode borderMode = BorderMode.Valid)
            : this(poolW, poolH, strideH, strideW,
          ConvUtils.PaddingFromBorderMode(poolH, borderMode),
          ConvUtils.PaddingFromBorderMode(poolW, borderMode))
        {
            m_borderMode = borderMode;
        }

        /// <summary>
        /// 
        /// </summary>
        public Variable Output { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Variable Input { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        public void Forward(NeuralNetStorage executor)
        {
            MaxPool2D.Forward(Input, Output, m_descriptor,
                m_switchX, m_switchY, executor);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        public void Backward(NeuralNetStorage executor)
        {
            MaxPool2D.Backward(Input, Output,
                m_switchX, m_switchY, executor);
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

            var batchSize = inputVariable.Dimensions[0];
            var fanOut = Output.DimensionOffSets[0]; // product of all dimensions except batch size.
            
            // store switches for x,y coordinates for where the max comes from, for each output neuron
            // only used during training.
            this.m_switchX = Enumerable.Range(0, batchSize).Select(v => new int[fanOut]).ToArray();
            this.m_switchY = Enumerable.Range(0, batchSize).Select(v => new int[fanOut]).ToArray();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(Variable input)
        {
            Input = input;

            var batchSize = input.Dimensions[0];
            var c = input.Dimensions[1];
            var h = input.Dimensions[2];
            var w = input.Dimensions[3];

            // compute output dimensions.
            var outC = c;
            var outH = ConvUtils.GetFilterGridLength(h, m_descriptor.PoolH, m_descriptor.StrideH, m_descriptor.PadH, m_borderMode);
            var outW = ConvUtils.GetFilterGridLength(w, m_descriptor.PoolW, m_descriptor.StrideW, m_descriptor.PadW, m_borderMode);

            Output = Variable.Create(batchSize, outC, outH, outW);
        }
    }
}
