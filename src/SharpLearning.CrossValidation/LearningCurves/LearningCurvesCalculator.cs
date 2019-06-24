using System;
using System.Collections.Generic;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.Samplers;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace SharpLearning.CrossValidation.LearningCurves
{
    /// <summary>
    /// Bias variance analysis calculator for constructing learning curves.
    /// Learning curves can be used to determine if a model has high bias or high variance.
    /// 
    /// Solutions for model with high bias:
    ///  - Add more features.
    ///  - Use a more sophisticated model
    ///  - Decrease regularization.
    /// Solutions for model with high variance
    ///  - Use fewer features.
    ///  - Use more training samples.
    ///  - Increase Regularization.
    /// </summary>
    public class LearningCurvesCalculator<TPrediction> : ILearningCurvesCalculator<TPrediction>
    {
        readonly ITrainingTestIndexSplitter<double> m_trainingValidationIndexSplitter;
        readonly double[] m_samplePercentages;
        readonly IMetric<double, TPrediction> m_metric;
        readonly IIndexSampler<double> m_indexedSampler;
        readonly int m_numberOfShufflesPrSample;
        readonly Random m_random;

        /// <summary>
        /// Bias variance analysis calculator for constructing learning curves.
        /// Learning curves can be used to determine if a model has high bias or high variance.
        /// </summary>
        /// <param name="trainingValidationIndexSplitter"></param>
        /// <param name="sampler">Type of shuffler to use when splitting data</param>
        /// <param name="metric">The error metric used</param>
        /// <param name="samplePercentages">A list of sample percentages determining the 
        /// training data used in each point of the learning curve</param>
        /// <param name="numberOfShufflesPrSample">How many times should the data be shuffled pr. calculated point</param>
        public LearningCurvesCalculator(ITrainingTestIndexSplitter<double> trainingValidationIndexSplitter,
            IIndexSampler<double> sampler, IMetric<double, TPrediction> metric, double[] samplePercentages, int numberOfShufflesPrSample = 5)
        {
            m_trainingValidationIndexSplitter = trainingValidationIndexSplitter ?? throw new ArgumentException(nameof(trainingValidationIndexSplitter));
            m_indexedSampler = sampler ?? throw new ArgumentException(nameof(sampler));
            m_metric = metric ?? throw new ArgumentNullException(nameof(metric));
            if (samplePercentages == null) { throw new ArgumentNullException("samplePercentages"); }
            if (samplePercentages.Length < 1) { throw new ArgumentException("SamplePercentages length must be at least 1"); }
            if (numberOfShufflesPrSample < 1) { throw new ArgumentNullException("numberOfShufflesPrSample must be at least 1"); }
            
            m_samplePercentages = samplePercentages;
            m_numberOfShufflesPrSample = numberOfShufflesPrSample;
            m_random = new Random(42);
        }

        /// <summary>
        /// Returns a list of BiasVarianceLearningCurvePoints for constructing learning curves.
        /// The points contain sample size, training score and validation score. 
        /// </summary>
        /// <param name="learnerFactory"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public List<LearningCurvePoint> Calculate(IIndexedLearner<TPrediction> learnerFactory,
            F64Matrix observations, double[] targets)
        {
            var trainingValidationIndices = m_trainingValidationIndexSplitter.Split(targets);
            
            return Calculate(learnerFactory, observations, targets,
                trainingValidationIndices.TrainingIndices,
                trainingValidationIndices.TestIndices);
        }

        /// <summary>
        /// Returns a list of BiasVarianceLearningCurvePoints for constructing learning curves.
        /// The points contain sample size, training score and validation score. 
        /// </summary>
        /// <param name="learner"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="trainingIndices">Indices that should be used for training</param>
        /// <param name="validationIndices">Indices that should be used for validation</param>
        /// <returns></returns>
        public List<LearningCurvePoint> Calculate(IIndexedLearner<TPrediction> learner,
            F64Matrix observations, double[] targets, int[] trainingIndices, int[] validationIndices)
        {
            var learningCurves = new List<LearningCurvePoint>();

            var validationTargets = targets.GetIndices(validationIndices);
            var validationPredictions = new TPrediction[validationTargets.Length];

            foreach (var samplePercentage in m_samplePercentages)
            {
                if (samplePercentage <= 0.0 || samplePercentage > 1.0)
                { 
                    throw new ArgumentException("Sample percentage must be larger than 0.0 and smaller than or equal to 1.0"); 
                }

                var sampleSize = (int)Math.Round(samplePercentage * (double)trainingIndices.Length);
                if (sampleSize <= 0)
                { 
                    throw new ArgumentException("Sample percentage " + samplePercentage + 
                        " too small for training set size " +trainingIndices.Length); 
                }

                var trainError = 0.0;
                var validationError = 0.0;

                var trainingPredictions = new TPrediction[sampleSize];

                for (int j = 0; j < m_numberOfShufflesPrSample; j++)
                {
                    var sampleIndices = m_indexedSampler.Sample(targets, sampleSize, trainingIndices);
                    var model = learner.Learn(observations, targets, sampleIndices);

                    for (int i = 0; i < trainingPredictions.Length; i++)
                    {
                        trainingPredictions[i] = model.Predict(observations.Row(sampleIndices[i]));
                    }

                    for (int i = 0; i < validationIndices.Length; i++)
                    {
                        validationPredictions[i] = model.Predict(observations.Row(validationIndices[i]));
                    }

                    var sampleTargets = targets.GetIndices(sampleIndices);
                    trainError += m_metric.Error(sampleTargets, trainingPredictions);
                    validationError += m_metric.Error(validationTargets, validationPredictions);

                    ModelDisposer.DisposeIfDisposable(model);
                }

                trainError = trainError / m_numberOfShufflesPrSample;
                validationError = validationError / m_numberOfShufflesPrSample;
                
                learningCurves.Add(new LearningCurvePoint(sampleSize,
                    trainError , validationError));
            }

            return learningCurves;
        }
    }
}
