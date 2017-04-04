using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Providers.DotNetOp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public class Conv2DLayer : ILayerNew
    {
        readonly Conv2DDescriptor m_descriptor;
        readonly BorderMode m_borderMode;

        Variable Weights;
        Variable Bias;
        Variable Im2Col;

        /// <summary>
        /// 2D Convolutional layer using GEMM implementation 
        /// based on: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
        /// and: https://arxiv.org/pdf/1410.0759.pdf
        /// </summary>
        /// <param name="filterCount">The number of filters</param>
        /// <param name="filterH">The height of the filters</param>
        /// <param name="filterW">The width of the filters</param>
        /// <param name="strideH">Controls the distance between each neighbouring filter for the height dimension (default is 1)</param>
        /// <param name="strideW">Controls the distance between each neighbouring filter for the width dimension (default is 1)</param>
        /// <param name="padH">Zero padding for the height dimension (default is 0)</param>
        /// <param name="padW">Zero padding for the width dimension (default is 0)</param>
        public Conv2DLayer(int filterCount, int filterH, int filterW,
            int strideH, int strideW,
            int padH, int padW)
        {
            m_descriptor = new Conv2DDescriptor(filterCount, filterH, filterW,
                strideH, strideW, padH, padW);

            m_borderMode = BorderMode.Undefined;
        }

        /// <summary>
        /// 2D Convolutional layer using GEMM implementation 
        /// based on: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
        /// and: https://arxiv.org/pdf/1410.0759.pdf
        /// </summary>
        /// <param name="filterCount">The number of filters</param>
        /// <param name="filterHeight">The height of the filters</param>
        /// <param name="filterWidth">The width of the filters</param>
        /// <param name="strideH">Controls the distance between each neighbouring filter for the height dimension (default is 1)</param>
        /// <param name="strideW">Controls the distance between each neighbouring filter for the width dimension (default is 1)</param>
        /// <param name="borderMode">Border mode of the convolutional operation. 
        /// This will set the width and height padding automatically based on the selected border mode: Valid, Same or Full (default is Valid)</param>
        public Conv2DLayer(int filterCount, int filterHeight, int filterWidth, 
            int strideH = 1, int strideW = 1,
            BorderMode borderMode = BorderMode.Valid)
            : this(filterCount, filterHeight, filterWidth, strideH, strideW,
                ConvUtils.PaddingFromBorderMode(filterHeight, borderMode),
                ConvUtils.PaddingFromBorderMode(filterWidth, borderMode))
        {
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
        /// <param name="storage"></param>
        /// <param name="training"></param>
        public void Forward(NeuralNetStorage storage, bool training = true)
        {
            Convolution.Forward(Input, Im2Col, m_descriptor,
               Weights, Bias, m_borderMode, Output, storage);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="storage"></param>
        public void Backward(NeuralNetStorage storage)
        {
            Convolution.Backward(Input, Im2Col, m_descriptor,
                Weights, Bias, m_borderMode, Output, storage);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="storage"></param>
        /// <param name="copyStorage"></param>
        /// <param name="layers"></param>
        public void Copy(Variable inputVariable, NeuralNetStorage storage, NeuralNetStorage copyStorage, List<ILayerNew> layers)
        {
            var copy = new Conv2DLayer(m_descriptor.FilterCount, m_descriptor.FilterH, m_descriptor.FilterW,
                m_descriptor.StrideH, m_descriptor.StrideW, m_descriptor.PadH, m_descriptor.PadW);

            copy.UpdateDimensions(inputVariable);

            copy.Weights = Weights.Copy();
            copyStorage.AssignTensor(copy.Weights, storage.GetTensor(Weights).Data.ToArray());

            copy.Bias = Bias.Copy();
            copyStorage.AssignTensor(copy.Bias, storage.GetTensor(Bias).Data.ToArray());            

            layers.Add(copy);
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
            var h = inputVariable.Dimensions[2];
            var w = inputVariable.Dimensions[3];

            var filterGridWidth = ConvUtils.GetFilterGridLength(w, m_descriptor.FilterW, 
                m_descriptor.StrideW, m_descriptor.PadW, m_borderMode);
            var filterGridHeight = ConvUtils.GetFilterGridLength(h, m_descriptor.FilterH,
                m_descriptor.StrideH, m_descriptor.PadW, m_borderMode);

            // Calculations of dimensions based on:
            // Nvidia, cuDNN: Efficient Primitives for Deep Learning: https://arxiv.org/pdf/1410.0759.pdf
            var filterCubeSize = c * filterGridWidth * filterGridHeight;

            var receptiveFieldSize = m_descriptor.FilterW * m_descriptor.FilterH;

            var fanIn = c * receptiveFieldSize;
            var fanOut = m_descriptor.FilterCount * receptiveFieldSize;
            var fans = new FanInFanOut(fanIn, fanOut);
            var distribution = WeightInitialization.GetWeightDistribution(initializtion, fans, random);

            Weights = Variable.CreateTrainable(m_descriptor.FilterCount, filterCubeSize);
            storage.AssignTensor(Weights, () => (float)distribution.Sample());

            Bias = Variable.CreateTrainable(m_descriptor.FilterCount);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(Variable input)
        {
            Input = input;

            var batchSize = Input.Dimensions[0];
            var c = Input.Dimensions[1];
            var h = Input.Dimensions[2];
            var w = Input.Dimensions[3];

            var filterGridWidth = ConvUtils.GetFilterGridLength(w, m_descriptor.FilterW,
                m_descriptor.StrideW, m_descriptor.PadW, m_borderMode);
            var filterGridHeight = ConvUtils.GetFilterGridLength(h, m_descriptor.FilterH,
                m_descriptor.StrideH, m_descriptor.PadW, m_borderMode);

            // Calculations of dimensions based on:
            // Nvidia, cuDNN: Efficient Primitives for Deep Learning: https://arxiv.org/pdf/1410.0759.pdf
            var filterCubeSize = c * filterGridWidth * filterGridHeight;
            var filterGridSize = filterGridWidth * filterGridHeight;

            Im2Col = Variable.Create(filterCubeSize, filterGridSize * batchSize);

            Output = Variable.Create(batchSize, m_descriptor.FilterCount,
                filterGridWidth, filterGridHeight);
        }
    }
}
