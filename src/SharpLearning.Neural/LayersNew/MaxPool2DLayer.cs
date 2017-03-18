using System;
using System.Linq;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class MaxPool2DLayer : ILayerNew
    {
        readonly int m_poolH;
        readonly int m_poolW;
        readonly int m_strideH;
        readonly int m_strideW;
        readonly int m_padH;
        readonly int m_padW;

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
            if (poolH < 1)
            { throw new ArgumentException($"filterH must be at least 1, was {poolH}"); }
            if (poolW < 1)
            { throw new ArgumentException($"filterW must be at least 1, was {poolW}"); }
            if (strideH < 1)
            { throw new ArgumentException($"strideH must be at least 1, was {strideH}"); }
            if (strideW < 1)
            { throw new ArgumentException($"strideW must be at least 1, was {strideW}"); }
            if (padH < 0)
            { throw new ArgumentException($"padH must be at least 0, was {padH}"); }
            if (padW < 0)
            { throw new ArgumentException($"padW must be at least 0, was {padW}"); }

            m_poolH = poolH;
            m_poolW = poolW;
            m_strideH = strideH;
            m_strideW = strideW;
            m_padH = padH;
            m_padW = padW;
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
        public void Forward(Executor executor)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        public void Backward(Executor executor)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="excecutor"></param>
        public void Initialize(Variable inputVariable, Executor excecutor)
        {
            Input = inputVariable;

            var batchSize = inputVariable.Dimensions[0];
            var c = inputVariable.Dimensions[1];
            var h = inputVariable.Dimensions[2];
            var w = inputVariable.Dimensions[3];

            // computed
            var outC = c;
            var outH = ConvUtils.GetFilterGridLength(h, m_poolH, m_strideH, m_padH, m_borderMode);
            var outW = ConvUtils.GetFilterGridLength(w, m_poolW, m_strideW, m_padW, m_borderMode);

            Output = Variable.Create(batchSize, outC, outH, outW);
            var fanOut = Output.DimensionOffSets[1]; // product of all dimensions except batch size.

            // store switches for x,y coordinates for where the max comes from, for each output neuron
            // only used during training.
            this.m_switchX = Enumerable.Range(0, batchSize).Select(v => new int[fanOut]).ToArray();
            this.m_switchY = Enumerable.Range(0, batchSize).Select(v => new int[fanOut]).ToArray();
        }
    }
}
