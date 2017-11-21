using System;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.CrossValidation.TimeSeries
{
    /// <summary>
    /// Time series cross-validation. Based on rolling validation using the original order of the data.
    /// Using the specified initial size of the training set, a model is trained. 
    /// The model predicts the first observation following the training data. 
    /// Following, this data point is included in the training and a new model is trained,
    /// which predict the next observation. This continous until all observations following the initial training size,
    /// has been validated.
    /// </summary>
    /// <typeparam name="TPrediction"></typeparam>
    public sealed class TimeSeriesCrossValidation<TPrediction>
    {
        readonly int m_initialTrainingSize;

        /// <summary>
        /// Time series cross-validation. Based on rolling validation.
        /// </summary>
        /// <param name="initialTrainingSize">The initial size of the training set.</param>
        public TimeSeriesCrossValidation(int initialTrainingSize)
        {
            if (initialTrainingSize <= 0) throw new ArgumentException($"initialTrainingSize much be larger than 0, was {initialTrainingSize}");
            m_initialTrainingSize = initialTrainingSize;
        }

        /// <summary>
        /// Time series cross-validation. Based on rolling validation using the original order of the data.
        /// Using the specified initial size of the training set, a model is trained. 
        /// The model predicts the first observation following the training data. 
        /// Following, this data point is included in the training and a new model is trained,
        /// which predict the next observation. This continous until all observations following the initial training size,
        /// has been validated.
        /// </summary>
        /// <param name="learner"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns>The validated predictions, following the initial training size</returns>
        public TPrediction[] Validate(IIndexedLearner<TPrediction> learner, F64Matrix observations, double[] targets)
        {
            if (observations.RowCount != targets.Length)
            { throw new ArgumentException($"observation row count {observations.RowCount} must match target length {targets.Length}"); }

            if (m_initialTrainingSize >= observations.RowCount)
            { throw new ArgumentException($"observation row count {observations.RowCount} is smaller than initial training size {m_initialTrainingSize}"); }

            var trainingIndices = Enumerable.Range(0, m_initialTrainingSize).ToArray();
            var predictionLength = targets.Length - trainingIndices.Length;
            var predictions = new TPrediction[predictionLength];

            var observation = new double[observations.ColumnCount];
            var currentObservationIndex = trainingIndices.Length;

            for (int i = 0; i < predictions.Length; i++)
            {
                var model = learner.Learn(observations, targets, trainingIndices);
                observations.Row(currentObservationIndex, observation);
                predictions[i] = model.Predict(observation);

                currentObservationIndex++;
                trainingIndices = Enumerable.Range(0, currentObservationIndex).ToArray();
            }

            return predictions;
        }

        /// <summary>
        /// Takes as input the original array of targets used as input for the validation,
        /// and returns the subset of targets corresponding to the validation prediction.
        /// </summary>
        /// <param name="targets">The original array of targets used as input for the validation</param>
        /// <returns>The subset of targets corresponding to the validation prediction</returns>
        public double[] GetValidationTargets(double[] targets)
        {
            return targets.Skip(m_initialTrainingSize).ToArray();
        }
    }
}
