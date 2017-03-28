using System;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Providers.DotNetOp;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class DenseLayer : ILayerNew
    {
        readonly int m_units;

        Variable Weights;
        Variable Bias;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="units"></param>
        public DenseLayer(int units)
        {
            if (units < 1) { throw new ArgumentException("HiddenLayer must have at least 1 hidden unit"); }
            m_units = units;
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
        /// <param name="training"></param>
        public void Forward(NeuralNetStorage executor, bool training = true)
        {
            Dense.Forward(Input, Weights, Bias,
                Output, executor);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        public void Backward(NeuralNetStorage executor)
        {
            Dense.Backward(Input, Weights, Bias,
                Output, executor);
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

            var fanIn = inputVariable.DimensionOffSets[0]; // product of all dimensions except batch size.
            var fanOut = m_units;

            var fans = new FanInFanOut(fanIn, fanOut);
            var distribution = WeightInitialization.GetWeightDistribution(initializtion, fans, random);
                        
            Weights = Variable.CreateTrainable(fans.FanIn, fans.FanOut);
            storage.AssignTensor(Weights, () => (float)distribution.Sample());

            Bias = Variable.CreateTrainable(fans.FanOut);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(Variable input)
        {
            Input = input;

            var batchSize = input.Dimensions[0];
            Output = Variable.Create(batchSize, m_units);
        }
    }
}
