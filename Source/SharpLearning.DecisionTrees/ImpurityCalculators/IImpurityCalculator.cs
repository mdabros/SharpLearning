using SharpLearning.Containers.Views;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    public interface IImpurityCalculator
    {
        void Init(double[] uniqueTargets, double[] targets, double[] weights, Interval1D interval);
        
        void UpdateInterval(Interval1D newInterval);
        void UpdateIndex(int newPosition);
        void Reset();

        double ImpurityImprovement(double impurity);

        double NodeImpurity();
        ChildImpurities ChildImpurities();

        double WeightedLeft { get; }
        double WeightedRight { get; }
        
        double LeafValue();
        Dictionary<double, double> LeafProbabilities();
    }
}
