using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.Shufflers;
using SharpLearning.Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.CrossValidation.CrossValidators
{
    /// <summary>
    /// Cross validation for evaluating how learning algorithms perform on unseen observations
    /// </summary>
    /// <typeparam name="TPrediction"></typeparam>
    public class CrossValidation<TPrediction>
    {
        readonly int m_crossValidationFolds;
        readonly ICrossValidationShuffler<double> m_shuffler;

        int[] m_workIndices = new int[0];
        int[] m_trainingIndices = new int[0];
        int[] m_holdoutIndices = new int[0];

        /// <summary>
        /// Cross validation for evaluating how learning algorithms perform on unseen observations
        /// </summary>
        /// <param name="shuffler">Shuffling strategy for the provided indices 
        /// before they are divided into the provided folds</param>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        public CrossValidation(ICrossValidationShuffler<double> shuffler, int crossValidationFolds)
        {
            if (shuffler == null) { throw new ArgumentNullException("shuffler"); }
            if (crossValidationFolds < 1) { throw new ArgumentException("CrossValidationFolds "); }

            m_shuffler = shuffler;
            m_crossValidationFolds = crossValidationFolds;
        }
        
        /// <summary>
        /// Returns an array of cross validated predictions
        /// </summary>
        /// <param name="learnerFactory"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public TPrediction[] CrossValidate(Func<IIndexedLearner<TPrediction>> learnerFactory, 
            F64Matrix observations, double[] targets)
        {
            var rows = targets.Length;
            if (m_crossValidationFolds > rows) 
            {
                throw new ArgumentException("Too few observations: " + rows + 
                " for number of cross validation folds: " + m_crossValidationFolds); 
            }
            
            Array.Resize(ref m_workIndices, targets.Length);
            for (int i = 0; i < targets.Length; i++)
            {
                m_workIndices[i] = i;
            }

            m_shuffler.Shuffle(m_workIndices, targets, m_crossValidationFolds);

            var crossValidatedPredictions = new TPrediction[rows];
            var countPrFold = rows / m_crossValidationFolds;
            var foldIndices = CreateIndicedFolds(m_workIndices, countPrFold);

            for (int i = 0; i < m_crossValidationFolds; i++)
            {
                AddCurrentIndices(foldIndices, rows, i);
                var model = learnerFactory().Learn(observations, targets, m_trainingIndices);
                var predictions = new TPrediction[m_holdoutIndices.Length];

                for (int l = 0; l < predictions.Length; l++)
                {
                    predictions[l] = model.Predict(observations.GetRow(l));  
                }

                for (int j = 0; j < m_holdoutIndices.Length; j++)
                {
                    crossValidatedPredictions[m_holdoutIndices[j]] = predictions[j];
                }
            }

            return crossValidatedPredictions;
        }

        void AddCurrentIndices(List<int[]> foldIndices, int totalLength, int i)
        {
            var holdoutIndices = foldIndices[i];
            var holdoutLength = holdoutIndices.Length;
            Array.Resize(ref m_holdoutIndices, holdoutLength);
            Array.Copy(holdoutIndices, m_holdoutIndices, holdoutLength);

            var trainingLength = totalLength - holdoutLength;
            Array.Resize(ref m_trainingIndices, trainingLength);
            
            var destinationIndex = 0;
            for (int j = 0; j < m_crossValidationFolds; j++)
            {               
                if (i == j)
                {
                    continue;
                }

                var indices = foldIndices[j];
                var length = indices.Length;

                Array.Copy(indices, 0, m_trainingIndices, destinationIndex, length);
                destinationIndex += length;
            }
        }

        private List<int[]> CreateIndicedFolds(int[] indices, int countPrFold)
        {
            var foldIndices = new List<int[]>();

            for (int i = 0; i < m_crossValidationFolds; i++)
            {
                if (i == m_crossValidationFolds - 1)
                {
                    var currentFoldIndices = indices.Skip(i * countPrFold).Take(indices.Length).ToArray();
                    foldIndices.Add(currentFoldIndices);
                }
                else
                {
                    var currentFoldIndices = indices.Skip(i * countPrFold).Take(countPrFold).ToArray();
                    foldIndices.Add(currentFoldIndices);
                }
            }
            return foldIndices;
        }
    }
}
