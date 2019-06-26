using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Ensemble.Learners
{
    /// <summary>
    /// Regression model selecting EnsembleLearner. 
    /// Trains several models and selects the best subset of models for the ensemble using iterative random selection.
    /// The selection of the best set of models is based on cross validation.
    /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
    /// </summary>
    public sealed class RegressionRandomModelSelectingEnsembleLearner : RegressionModelSelectingEnsembleLearner
    {
        /// <summary>
        /// Regression model selecting EnsembleLearner. 
        /// Trains several models and selects the best subset of models for the ensemble using iterative random selection.
        /// The selection of the best set of models is based on cross validation. 
        /// Default is 5-fold RandomCrossValidation and minimization of mean square error and mean is used to combine the models.
        /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="numberOfModelsToSelect">Number of models to select</param>
        /// <param name="iterations">Number of iterations to random select model combinations.</param> 
        public RegressionRandomModelSelectingEnsembleLearner(
            IIndexedLearner<double>[] learners, 
            int numberOfModelsToSelect, 
            int iterations=50)
            : this(learners, numberOfModelsToSelect, 
                new RandomCrossValidation<double>(5, 42), 
                new MeanRegressionEnsembleStrategy(),
                new MeanSquaredErrorRegressionMetric())
        {
        }


        /// <summary>
        /// Regression model selecting EnsembleLearner. 
        /// Trains several models and selects the best subset of models for the ensemble using iterative random selection.
        /// The selection of the best set of models is based on cross validation. 
        /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="numberOfModelsToSelect">Number of models to select</param>
        /// <param name="crossValidation">Cross validation method</param>
        /// <param name="ensembleStrategy">Strategy for ensembling models</param>
        /// <param name="metric">Metric to minimize</param>
        /// <param name="iterations">Number of iterations to random select model combinations.</param> 
        /// <param name="selectWithReplacement">If true the same model can be selected multiple times.</param>
        /// <param name="seed"></param>
        public RegressionRandomModelSelectingEnsembleLearner(
            IIndexedLearner<double>[] learners, 
            int numberOfModelsToSelect,
            ICrossValidation<double> crossValidation, 
            IRegressionEnsembleStrategy ensembleStrategy, 
            IMetric<double, double> metric, 
            int iterations = 50, 
            bool selectWithReplacement = true, 
            int seed = 42)
            : base(learners, crossValidation, ensembleStrategy, 
                new RandomRegressionEnsembleSelection(metric, ensembleStrategy, numberOfModelsToSelect, 
                    iterations, selectWithReplacement, seed))
        {
        }
    }
}
