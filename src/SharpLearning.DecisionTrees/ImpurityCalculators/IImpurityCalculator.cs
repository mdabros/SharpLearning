using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    /// <summary>
    /// Interface for impurity calculators
    /// </summary>
    public interface IImpurityCalculator
    {
        /// <summary>
        /// Initialize the calculator with targets, weights and work interval
        /// </summary>
        /// <param name="uniqueTargets">The available target values</param>
        /// <param name="targets"></param>
        /// <param name="weights"></param>
        /// <param name="interval"></param>
        void Init(double[] uniqueTargets, double[] targets, double[] weights, Interval1D interval);
        
        /// <summary>
        /// Update the work interval.
        /// </summary>
        /// <param name="newInterval"></param>
        void UpdateInterval(Interval1D newInterval);
        
        /// <summary>
        /// Update the split index
        /// </summary>
        /// <param name="newPosition"></param>
        void UpdateIndex(int newPosition);
        
        /// <summary>
        /// Reset the calculation within the current work interval
        /// </summary>
        void Reset();

        /// <summary>
        /// Return the impurity improvement
        /// </summary>
        /// <param name="impurity"></param>
        /// <returns></returns>
        double ImpurityImprovement(double impurity);

        /// <summary>
        /// Returns the current node impurity
        /// </summary>
        /// <returns></returns>
        double NodeImpurity();

        /// <summary>
        /// Returns the current child impurities
        /// </summary>
        /// <returns></returns>
        ChildImpurities ChildImpurities();

        /// <summary>
        /// Returns the weighted size of the current left split
        /// </summary>
        double WeightedLeft { get; }

        /// <summary>
        /// Returns the weighted size of the current right split
        /// </summary>
        double WeightedRight { get; }
        
        /// <summary>
        /// Calculates the leaf value based on the current work interval
        /// </summary>
        /// <returns></returns>
        double LeafValue();

        /// <summary>
        /// Gets the unique target names
        /// </summary>
        double[] TargetNames { get; }

        /// <summary>
        /// Calculates the probabilities based in the current work interval. 
        /// Note that LeafProbabilities are only valid for classification impurity calculators.
        /// Regression impurity calculators will return and empty result. The orders of the probabilities 
        /// is the same as TargetNames.
        /// </summary>
        /// <returns></returns>
        double[] LeafProbabilities();
    }
}
