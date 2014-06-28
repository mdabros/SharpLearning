using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.Shufflers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.CrossValidation
{
    /// <summary>
    /// Cross validation for evaluating how learning algorithms perform on unseen observations
    /// </summary>
    /// <typeparam name="TOut"></typeparam>
    public class CrossValidation<TOut, TTarget>
    {
        readonly CrossValidationLearner<TOut, TTarget> m_modelLearner;
        readonly int m_crossValidationFolds;
        readonly ICrossValidationShuffler<TTarget> m_shuffler;

        int[] m_workIndices = new int[0];
        int[] m_trainingIndices = new int[0];
        int[] m_holdoutIndices = new int[0];

        /// <summary>
        /// Cross validation for evaluating how learning algorithms perform on unseen observations
        /// </summary>
        /// <param name="modelLearner">The func should provide a learning algorithm 
        /// that returns a model predicting multiple observations</param>
        /// <param name="shuffler">Shuffling strategy for the provided indices 
        /// before they are divided into the provided folds</param>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        public CrossValidation(CrossValidationLearner<TOut, TTarget> modelLearner,
                                    ICrossValidationShuffler<TTarget> shuffler, int crossValidationFolds)
        {
            if (modelLearner == null) { throw new ArgumentNullException("trainer"); }
            if (shuffler == null) { throw new ArgumentNullException("shuffler"); }
            if (crossValidationFolds < 1) { throw new ArgumentException("CrossValidationFolds "); }

            m_modelLearner = modelLearner;
            m_shuffler = shuffler;
            m_crossValidationFolds = crossValidationFolds;
        }
        
        /// <summary>
        /// Returns an array of cross validated predictions
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public TOut[] CrossValidate(F64Matrix observations, TTarget[] targets)
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

            var crossValidatedPredictions = new TOut[rows];
            var countPrFold = rows / m_crossValidationFolds;
            var foldIndices = CreateIndicedFolds(m_workIndices, countPrFold);

            for (int i = 0; i < m_crossValidationFolds; i++)
            {
                AddCurrentIndices(foldIndices, rows, i);
                var model = m_modelLearner(observations, targets, m_trainingIndices);
                var predictions = model(observations, m_holdoutIndices);

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
