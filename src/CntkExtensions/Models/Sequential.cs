using System;
using System.Collections.Generic;
using CNTK;

namespace CntkExtensions.Models
{
    public class Sequential
    {
        // creators.
        List<Func<Function, Function>> m_layersCreators;
        Func<IList<Parameter>, Learner> LearnerCreator;
        Func<Variable, Variable, Function> LossCreator;
        Func<Variable, Variable, Function> MetricCreator;

        public Function Network;

        public Sequential()
        {
            m_layersCreators = new List<Func<Function, Function>>();
        }

        public void Add(Func<Function, Function> layerCreator)
        {
            m_layersCreators.Add(layerCreator);
        }

        public void Compile(Func<IList<Parameter>, Learner> learnerCreator,
            Func<Variable, Variable, Function> lossCreator,
            Func<Variable, Variable, Function> metricCreator)
        {
            LearnerCreator = learnerCreator;
            LossCreator = lossCreator;
            MetricCreator = metricCreator;
        }

        private void CreateNetwork(Variable input)
        {
            // Setup actual network function.
            Network = input;
            foreach (var creator in m_layersCreators)
            {
                Network = creator(Network);
            }
        }
    }
}
