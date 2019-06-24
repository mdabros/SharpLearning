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
        readonly int m_maxTrainingSetSize;
        readonly int m_retrainInterval;

        /// <summary>
        /// Time series cross-validation. Based on rolling validation.
        /// </summary>
        /// <param name="initialTrainingSize">The initial size of the training set.</param>
        /// <param name="maxTrainingSetSize">The maximum size of the training set. Default is 0, which indicate no maximum size, 
        /// resulting in an expanding training interval. If a max is chosen, and the max size is reached, 
        /// this will result in a sliding training interval, moving forward in time, 
        /// always using the data closest to the test period as training data. </param>
        /// <param name="retrainInterval">How often should the model be retrained. Default is 1, which will retrain the model at all time steps.
        /// Setting the interval to 5 will retrain the model at every fifth time step and use the current model for all time steps in between.</param>
        public TimeSeriesCrossValidation(int initialTrainingSize, int maxTrainingSetSize = 0, int retrainInterval = 1)
        {
            if (initialTrainingSize <= 0)
            {
                throw new ArgumentException($"{nameof(initialTrainingSize)} " + 
                    $"much be larger than 0, was {initialTrainingSize}");
            }

            if (maxTrainingSetSize < 0)
            {
                throw new ArgumentException($"{nameof(maxTrainingSetSize)} " + 
                    $"much be larger than 0, was {maxTrainingSetSize}");
            }

            if ((maxTrainingSetSize != 0) && (initialTrainingSize > maxTrainingSetSize))
            {
                throw new ArgumentException($"{nameof(initialTrainingSize)} = {initialTrainingSize} " + 
                    $"is larger than {nameof(maxTrainingSetSize)} = {maxTrainingSetSize}");
            }

            if (retrainInterval < 1)
            {
                throw new ArgumentException($"{nameof(retrainInterval)} much be larger than 1, " + 
                    $"was {retrainInterval}");
            }
            
            m_initialTrainingSize = initialTrainingSize;
            m_maxTrainingSetSize = maxTrainingSetSize;
            m_retrainInterval = retrainInterval;
        }

        /// <summary>
        /// Time series cross-validation. Based on rolling validation using the original order of the data.
        /// Using the specified initial size of the training set, a model is trained. 
        /// The model predicts the first observation following the training data. 
        /// Following, this data point is included in the training and a new model is trained,
        /// which predict the next observation. This continuous until all observations following the initial training size,
        /// has been validated.
        /// </summary>
        /// <param name="learner"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns>The validated predictions, following the initial training size</returns>
        public TPrediction[] Validate(IIndexedLearner<TPrediction> learner, F64Matrix observations, double[] targets)
        {
            if (observations.RowCount != targets.Length)
            {
                throw new ArgumentException($"observation row count {observations.RowCount} " + 
                    $"must match target length {targets.Length}");
            }

            if (m_initialTrainingSize >= observations.RowCount)
            {
                throw new ArgumentException($"observation row count {observations.RowCount} " + 
                    $"is smaller than initial training size {m_initialTrainingSize}");
            }

            var trainingIndices = Enumerable.Range(0, m_initialTrainingSize).ToArray();
            var predictionLength = targets.Length - trainingIndices.Length;
            var predictions = new TPrediction[predictionLength];

            var observation = new double[observations.ColumnCount];
            var lastTrainingIndex = trainingIndices.Last();

            var model = learner.Learn(observations, targets, trainingIndices);

            for (int i = 0; i < predictions.Length; i++)
            {
                // Only train a new model at each retrain interval.
                if((m_retrainInterval == 1 || i % m_retrainInterval == 0) && i != 0)
                {
                    model = learner.Learn(observations, targets, trainingIndices);
                }

                var predictionIndex = lastTrainingIndex + 1;
                observations.Row(predictionIndex, observation);
                predictions[i] = model.Predict(observation);

                lastTrainingIndex++;
                
                // determine start index and length of the training period, if maxTrainingSetSize is specified. 
                var startIndex = m_maxTrainingSetSize != 0 ? 
                    Math.Max(0, (lastTrainingIndex + 1) - m_maxTrainingSetSize) : 0;

                var length = m_maxTrainingSetSize != 0 ? 
                    Math.Min(m_maxTrainingSetSize, lastTrainingIndex) : lastTrainingIndex;

                trainingIndices = Enumerable.Range(startIndex, length).ToArray();

                ModelDisposer.DisposeIfDisposable(model);
            }

            return predictions;
        }

        /// <summary>
        /// Takes as input the original array of targets used as input for the validation,
        /// and returns the subset of targets corresponding to the validation predictions.
        /// </summary>
        /// <param name="targets">The original array of targets used as input for the validation</param>
        /// <returns>The subset of targets corresponding to the validation predictions</returns>
        public double[] GetValidationTargets(double[] targets)
        {
            return targets.Skip(m_initialTrainingSize).ToArray();
        }

        /// <summary>
        /// Takes as input the original array of targets used as input for the validation,
        /// and returns the indices used for the validation predictions.
        /// </summary>
        /// <param name="targets">The original array of targets used as input for the validation</param>
        /// <returns>The subset of indices used for the validation predictions</returns>
        public int[] GetValidationIndices(double[] targets)
        {
            return Enumerable.Range(0, targets.Length)
                .Skip(m_initialTrainingSize).ToArray();
        }
    }
}
