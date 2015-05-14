using SharpLearning.GradientBoost.LossFunctions;

namespace SharpLearning.GradientBoost.Learners
{
    /// <summary>
    /// Least squares (LS) regression gradient boost learner based on 
    /// http://statweb.stanford.edu/~jhf/ftp/trebst.pdf
    /// A series of regression trees are fitted stage wise on the residuals of the previous stage.
    /// The resulting models are ensembled together using addition.
    /// </summary>
    public sealed class RegressionLeastSquaresGradientBoostLearner : RegressionGradientBoostLearner
    {
        /// <summary>
        ///  Least squares (LS) regression gradient boost learner. 
        ///  A series of regression trees are fitted stage wise on the residuals of the previous stage
        /// </summary>
        /// <param name="iterations">The number of iterations or stages</param>
        /// <param name="learningRate">How much each iteration should contribute with</param>
        /// <param name="maximumTreeDepth">The maximum depth of the tree models</param>
        /// <param name="maximumLeafCount">The maximum leaf count of the tree models</param>
        /// <param name="minimumSplitSize">minimum node split size in the trees 1 is default</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="subSampleRatio">ratio of observations sampled at each iteration. Default is 1.0. 
        /// If below 1.0 the algorithm changes to stochastic gradient boosting. 
        /// This reduces variance in the ensemble and can help ounter overfitting</param>
        /// <param name="numberOfFeaturesPrSplit">Number of features used at each split in each tree. 0 means Sqrt(of availible features)</param>
        public RegressionLeastSquaresGradientBoostLearner(int iterations = 100, double learningRate = 0.1, int maximumTreeDepth = 3, 
            int maximumLeafCount=2000, int minimumSplitSize = 1, double minimumInformationGain = 0.000001, double subSampleRatio = 1.0, int numberOfFeaturesPrSplit = 0)
            : base(new LeastSquaresLossFunction(learningRate), iterations, maximumTreeDepth,
                    maximumLeafCount, minimumSplitSize, minimumInformationGain, subSampleRatio, numberOfFeaturesPrSplit)
        {
        }
    }
}
