using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CNTK;

namespace SharpLearning.Neural.Cntk
{
    public class CntkNeuralNetLearner
    {
        readonly Function m_network;
        readonly DeviceDescriptor m_device;

        readonly Learner m_optimizer;
        readonly Function m_loss;
        readonly Function m_metric;

        readonly int m_minibatchIterations;
        readonly uint m_minibatchSize;

        public CntkNeuralNetLearner(Function network, DeviceDescriptor device,
            int miniBatchIterations, int minibatchSize,
            Learner optimizer, Function loss, Function metric)
        {
            if (network == null) { throw new ArgumentNullException("network"); }
            if (device == null) { throw new ArgumentNullException("device"); }
            if (optimizer == null) throw new ArgumentNullException(nameof(optimizer));
            if (loss == null) throw new ArgumentNullException(nameof(loss));
            if (metric == null) throw new ArgumentNullException(nameof(metric));

            m_network = network;
            m_device = device;
            m_optimizer = optimizer;
            m_loss = loss;
            m_metric = metric;
            m_minibatchIterations = miniBatchIterations;
            m_minibatchSize = (uint)minibatchSize; // add check
        }

        public CntkNeuralNetModel Learn(MinibatchSource minibatchSource, Variable targetVariable)
        {
            var inputVariable = m_network.Arguments[0];
            var inputShape = m_network.Arguments[0].Shape.Dimensions;
            var inputDimensions = inputShape.Aggregate((d1, d2) => d1 * d2);

            // Instantiate progress writers.
            var trainingProgressOutputFreq = 100;

            var featureStreamInfo = minibatchSource.StreamInfo(inputVariable);
            var labelStreamInfo = minibatchSource.StreamInfo(targetVariable);

            // Train model.
            var trainer = Trainer.CreateTrainer(m_network, m_loss, m_metric, new List<Learner> { m_optimizer });

            for (int i = 0; i < m_minibatchIterations; i++)
            {
                var mb = minibatchSource.GetNextMinibatch(m_minibatchSize, m_device);
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { inputVariable, mb[featureStreamInfo] },
                    { targetVariable, mb[labelStreamInfo] }
                };
                trainer.TrainMinibatch(arguments, m_device);

                if (((i + 1) % trainingProgressOutputFreq) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
                {
                    var trainLossValue = trainer.PreviousMinibatchLossAverage();
                    var evaluationValue = trainer.PreviousMinibatchEvaluationAverage();
                    Trace.WriteLine($"Minibatch: {i + 1} Loss = {trainLossValue:F16}, Metric = {evaluationValue:F16}");
                }
            }

            return new CntkNeuralNetModel(m_network.Clone(ParameterCloningMethod.Clone), m_device); 
        }
    }
}
