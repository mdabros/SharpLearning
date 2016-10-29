using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Models;
using SharpLearning.Neural.Optimizers;
using SharpLearning.Neural.TargetEncoders;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.Neural.Learners
{
    /// <summary>
    /// Base NeuralNet Learner
    /// </summary>
    public class NeuralNetLearner
    {
        readonly int[] m_hiddenLayerSizes;

        readonly IActivation m_hiddenActiviation;
        readonly IActivation m_outputActiviation;
        
        /// <summary>
        /// func to get hidden activation
        /// </summary>
        protected readonly Func<IActivation> m_hiddenActiviationFunc;

        /// <summary>
        /// func to get output activation
        /// </summary>
        protected readonly Func<IActivation> m_outputActiviationFunc;

        readonly INeuralNetOptimizer m_optimizer;
        readonly ITargetEncoder m_targetEncoder;
        readonly ILoss m_loss;
        readonly float m_l2reguralization;
        readonly int m_batchSize;
        readonly int m_maxIterations;
        readonly bool m_shuffle;
        readonly Random m_random;
        readonly Random m_dropoutRandom;
        readonly float m_tol;
        readonly bool m_dropout;
        readonly List<double> m_dropOutValues;

        int m_n_iter;
        int m_t;
        int m_n_outputs;
        int m_n_layers;
        int m_no_improvement_count;

        /// <summary>
        /// neural net coefs
        /// </summary>
        protected List<Matrix<float>> m_coefs;
        
        /// <summary>
        /// neural net intercepts
        /// </summary>
        protected List<Vector<float>> m_intercepts;

        Dictionary<int, Matrix<float>> m_dropMasks;
        Dictionary<int, float[]> m_dropArrays;

        double m_best_loss;
        double m_currentLoss;

        /// <summary>
        /// Base NeuralNet Learner
        /// </summary>
        /// <param name="hiddenLayers">Hidden layers. The layers is initializes in the order they appear in the array </param>
        /// <param name="activiation">The type of activation used in the hidden layers</param>
        /// <param name="loss">Loss to minimize</param>
        /// <param name="targetEncoder">Encodes targets to the format suitable for neural net</param>
        /// <param name="outputActivation">Output activation function</param>
        /// <param name="optimizer">Optimizer for running the learning process</param>
        /// <param name="maxIterations">The maximum number of iterations before termination. </param>
        /// <param name="batchSize">Batch size for mini-batch stochastic gradient descent.</param>
        /// <param name="l2regularization">L2 reguralization term</param>
        /// <param name="inputDropOut">Input dropout percentage. The percentage of units randomly omitted during training.
        /// This is a reguralizatin methods for reducing overfitting. Recommended value is 0.1 and range should be between 0.0 and 0.3</param>
        /// <param name="shuffle">Decides if the observations should be shuffled between each iteration. </param>
        /// <param name="seed">Seed for random initialization of weights.</param>
        /// <param name="tol">Tolerence for the optimization. If the training loss has not improved be tol 
        /// in two consequitive iterations. The optimization terminates</param>
        public NeuralNetLearner(HiddenLayer[] hiddenLayers, IActivation activiation, ILoss loss, ITargetEncoder targetEncoder, IActivation outputActivation,
            INeuralNetOptimizer optimizer, int maxIterations = 200, int batchSize = 100, double l2regularization = 0.0001, double inputDropOut = 0.0,
            bool shuffle = true, int seed = 42, double tol = 1e-4)
            : this(hiddenLayers, () => activiation, loss, targetEncoder, () => outputActivation, optimizer, maxIterations, batchSize, l2regularization, inputDropOut,
                shuffle, seed, tol)
        {
        }

        /// <summary>
        /// Base NeuralNet Learner
        /// </summary>
        /// <param name="hiddenLayers">Hidden layers. The layers is initializes in the order they appear in the array </param>
        /// <param name="activiation">The type of activation used in the hidden layers</param>
        /// <param name="loss">Loss to minimize</param>
        /// <param name="targetEncoder">Encodes targets to the format suitable for neural net</param>
        /// <param name="outputActivation">Output activation function</param>
        /// <param name="optimizer">Optimizer for running the learning process</param>
        /// <param name="maxIterations">The maximum number of iterations before termination. </param>
        /// <param name="batchSize">Batch size for mini-batch stochastic gradient descent.</param>
        /// <param name="l2regularization">L2 reguralization term</param>
        /// <param name="inputDropOut">Input dropout percentage. The percentage of units randomly omitted during training.
        /// This is a reguralizatin methods for reducing overfitting. Recommended value is 0.1 and range should be between 0.0 and 0.3</param>
        /// <param name="shuffle">Decides if the observations should be shuffled between each iteration. </param>
        /// <param name="seed">Seed for random initialization of weights.</param>
        /// <param name="tol">Tolerence for the optimization. If the training loss has not improved be tol 
        /// in two consequitive iterations. The optimization terminates</param>
        public NeuralNetLearner(HiddenLayer[] hiddenLayers, Func<IActivation> activiation, ILoss loss, ITargetEncoder targetEncoder, Func<IActivation> outputActivation,
            INeuralNetOptimizer optimizer, int maxIterations = 200, int batchSize = 100, double l2regularization = 0.0001, double inputDropOut = 0.0,
            bool shuffle = true, int seed = 42, double tol = 1e-4)
        {
            if (hiddenLayers == null) { throw new ArgumentNullException("hiddenLayerSizes"); }
            if (activiation == null) { throw new ArgumentNullException("activiation"); }
            if (loss == null) { throw new ArgumentNullException("loss"); }
            if (targetEncoder == null) { throw new ArgumentNullException("targetEncoder"); }
            if (outputActivation == null) { throw new ArgumentNullException("outputActivation"); }
            if (optimizer == null) { throw new ArgumentNullException("optimizer"); }
            if (inputDropOut < 0.0 || inputDropOut >= 1.0) { throw new ArgumentException("InputDropOut must be below 1.0 and at least 0.0"); }
            if (maxIterations < 1 ) { throw new ArgumentException("maxIterations must be at least 1"); }
            if (batchSize < 1) { throw new ArgumentException("batchSize must be at least 1"); }
            if (l2regularization < 0.0) { throw new ArgumentException("l2regularization must be at least 0"); }
            if (tol < 0.0) { throw new ArgumentException("tol must be at least 0"); }

            m_hiddenActiviation = activiation();
            if (m_hiddenActiviation == null) { throw new ArgumentNullException("activiation"); }
            m_outputActiviation = outputActivation();
            if (m_outputActiviation == null) { throw new ArgumentNullException("outputActivation"); }

            m_hiddenLayerSizes = new int[hiddenLayers.Length];
            m_hiddenActiviationFunc = activiation;
            m_outputActiviationFunc = outputActivation;
            m_optimizer = optimizer;
            m_loss = loss;
            m_targetEncoder = targetEncoder;
            m_l2reguralization = (float)l2regularization;
            m_batchSize = batchSize;
            m_maxIterations = maxIterations;
            m_shuffle = shuffle;
            m_random = new Random(seed);
            m_tol = (float)tol;

            SetupLinerAlgebraProvider();
            m_dropoutRandom = new Random(232);
            m_dropOutValues = new List<double>();

            // setup hidden layer sizes
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                m_hiddenLayerSizes[i] = hiddenLayers[i].Units;
            }

            // setup dropout values
            m_dropOutValues.Add(inputDropOut); // input dropout
            foreach (var layer in hiddenLayers)
            {
                m_dropOutValues.Add(layer.DropOut);
            }

            m_dropout = m_dropOutValues.Any(f => f > 0.0);
        }

        void SetupLinerAlgebraProvider()
        {
            if (Control.TryUseNativeMKL())
            { }
            else if (Control.TryUseNativeOpenBLAS())
            { }
            else
            {
                Control.UseManaged();
                Control.UseMultiThreading();
            }
        }

        /// <summary>
        /// Learns a neural net model
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="validationObservations"></param>
        /// <param name="validationTargets"></param>
        /// <param name="earlyStoppingRounds"></param>
        /// <param name="earlyStoppingFunc"></param>
        /// <returns></returns>
        protected NeuralNetModel BaseLearn(F64Matrix observations, double[] targets,
            F64Matrix validationObservations, double[] validationTargets, int earlyStoppingRounds,
            Func<F64Matrix, double[], double> earlyStoppingFunc)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return BaseLearn(observations, targets, indices, validationObservations, 
                validationTargets, earlyStoppingRounds, earlyStoppingFunc);
        }

        /// <summary>
        /// Learns a neural net model
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <param name="validationObservations"></param>
        /// <param name="validationTargets"></param>
        /// <param name="earlyStoppingRounds"></param>
        /// <param name="earlyStoppingFunc"></param>
        /// <returns></returns>
        protected NeuralNetModel BaseLearn(F64Matrix observations, double[] targets, int[] indices,
            F64Matrix validationObservations, double[] validationTargets, int earlyStoppingRounds, 
            Func<F64Matrix, double[], double> earlyStoppingFunc)
        {
            var n_samples = indices.Length;
            var n_features = observations.GetNumberOfColumns();

            var oneOfNTargets = m_targetEncoder.Encode(targets);
            m_n_outputs = oneOfNTargets.ColumnCount;

            var layerUnits = new List<int> { n_features };
            layerUnits.AddRange(m_hiddenLayerSizes);
            layerUnits.Add(m_n_outputs);

            // initialize all layes
            Initialize(oneOfNTargets, layerUnits);

            var activations = new List<Matrix<float>>();
            var activationsTemp = new List<Matrix<float>>();
            var deltas = new List<Matrix<float>>();

            for (int i = 0; i < layerUnits.Count; i++)
            {
                var fanOut = layerUnits[i];
                activations.Add(Matrix<float>.Build.Dense(m_batchSize, fanOut));
                activationsTemp.Add(Matrix<float>.Build.Dense(m_batchSize, fanOut));

                if (i != 0)
                {
                    deltas.Add(Matrix<float>.Build.Dense(m_batchSize, fanOut));
                }
            }

            // initialize dropout
            if (m_dropout)
            {
                m_dropArrays = new Dictionary<int, float[]>();
                m_dropMasks = new Dictionary<int, Matrix<float>>();

                for (int i = 0; i < layerUnits.Count - 1; i++)
                {
                    if (m_dropOutValues[i] > 0.0)
                    {
                        var array = Enumerable.Range(0, m_batchSize * layerUnits[i]).Select(v => DecideDropOut(i)).ToArray();
                        var matrix = Matrix<float>.Build.Dense(m_batchSize, layerUnits[i], array);
                        m_dropMasks.Add(i, matrix);
                        m_dropArrays.Add(i, array);
                    }
                }
            }

            var intercepts_grads = layerUnits.Skip(1)
                .Select(i => Vector<float>.Build.Dense(i)).ToList();

            var coefs_grads = new List<Matrix<float>>();
            for (int i = 0; i < layerUnits.Count - 1; i++)
            {
                var fanIn = layerUnits[i];
                var fanOut = layerUnits[i + 1];
                coefs_grads.Add(Matrix<float>.Build.Dense(fanIn, fanOut));
            }

            var learningIndices = indices.ToArray();
            var numberOfBatches = n_samples / m_batchSize;
            var batchTargets = Matrix<float>.Build.Dense(m_batchSize, (int)oneOfNTargets.ColumnCount);

            // initialize optimizer
            m_optimizer.SetParameters(m_coefs, m_intercepts);


            var timer = new Stopwatch();

            // early stopping variables
            var iterationsUsed = 0;
            var bestValidationError = double.MaxValue;
            List<Matrix<float>> bestCoefs = null;
            List<Vector<float>> bestIntercepts = null;

            var useEarlyStopping = validationObservations != null && validationTargets != null && earlyStoppingRounds != 0 && earlyStoppingFunc != null;
            if (useEarlyStopping)
            {
                bestCoefs = new List<Matrix<float>>();
                bestIntercepts = new List<Vector<float>>();
            }

            // train using stochastic gradient descent
            for (int iteration = 0; iteration < m_maxIterations; iteration++)
            {
                timer.Restart();
                var accumulatedLoss = 0.0;

                if (m_shuffle)
                {
                    learningIndices.Shuffle(m_random);
                }

                for (int i = 0; i < numberOfBatches; i++)
                {
                    var workIndices = learningIndices.Skip(i * m_batchSize).Take(m_batchSize).ToArray();

                    if (workIndices.Length != m_batchSize)
                    {
                        continue; // only train with full batch size
                    }

                    CopyBatchTargets(oneOfNTargets, batchTargets, workIndices);
                    CopyBatch(observations, activations[0], workIndices);

                    var batchLoss = BackPropagate(activations[0], batchTargets, activations, activationsTemp,
                        deltas, coefs_grads, intercepts_grads);

                    accumulatedLoss += batchLoss * m_batchSize;

                    m_optimizer.UpdateParameters(coefs_grads, intercepts_grads);
                }

                timer.Stop();

                m_n_iter += 1;
                m_currentLoss = accumulatedLoss / (double)indices.Length;
                m_t += n_samples;

                if (useEarlyStopping && iteration % earlyStoppingRounds == 0)
                {
                    var validationError = earlyStoppingFunc(validationObservations, validationTargets);
                    Trace.WriteLine("Iteration: " + (iteration + 1) + " Loss: " + m_currentLoss + " Validation Loss: " + validationError + " Time (ms): " + timer.ElapsedMilliseconds);

                    if (validationError > bestValidationError)
                    {
                        Trace.WriteLine("Validation error has not improved during the last " + earlyStoppingRounds + " iterations. Stopping");
                        break;
                    }
                    else
                    {
                        iterationsUsed = iteration;
                        bestValidationError = validationError;
                        CopyCurrentBestParameters(bestCoefs, bestIntercepts);
                    }
                }
                else
                {
                    Trace.WriteLine("Iteration: " + (iteration + 1) + " Loss: " + m_currentLoss + " Time (ms): " + timer.ElapsedMilliseconds);
                }


                UpdateNoImprovement();

                m_optimizer.IterationEnds(m_t);

                if (double.IsNaN(m_currentLoss))
                {
                    Trace.WriteLine("Training loss is NaN, stopping");
                    iterationsUsed = iteration;
                    break;
                }

                if(!useEarlyStopping)
                {
                    if (m_no_improvement_count > 2)
                    {
                        Trace.WriteLine("Training loss did not improve more than tol=" + m_tol + " for two iterations");

                        if (m_optimizer.TriggerStopping())
                        {
                            iterationsUsed = iteration;
                            break;
                        }
                        else
                        {
                            m_no_improvement_count = 0;
                        }
                    }
                }

                if (iteration == (m_maxIterations - 1))
                {
                    Trace.WriteLine("Max iterations reached");
                    iterationsUsed = iteration;
                }
            }

            if(useEarlyStopping)
            {
                return new NeuralNetModel(bestCoefs,  bestIntercepts,
                    m_hiddenActiviationFunc(), m_outputActiviationFunc(), iterationsUsed);
            }
            else
            {
                return new NeuralNetModel(m_coefs.ToList(), m_intercepts.ToList(),
                    m_hiddenActiviationFunc(), m_outputActiviationFunc(), iterationsUsed);
            }
        }

        void CopyCurrentBestParameters(List<Matrix<float>> bestCoefs, List<Vector<float>> bestIntercepts)
        {
            bestCoefs.Clear();
            bestIntercepts.Clear();

            for (int i = 0; i < m_coefs.Count; i++)
            {
                // copy current best coefs
                var copyCoef = Matrix<float>.Build.Dense(m_coefs[i].RowCount, m_coefs[i].ColumnCount);
                m_coefs[i].CopyTo(copyCoef);
                bestCoefs.Add(copyCoef);

                // copy current best inercepts
                var copyIntercept = Vector<float>.Build.Dense(m_intercepts[i].Count);
                m_intercepts[i].CopyTo(copyIntercept);
                bestIntercepts.Add(copyIntercept);
            }
        }

        void UpdateNoImprovement()
        {
            if (m_currentLoss > m_best_loss - m_tol)
            {
                m_no_improvement_count++;
            }
            else
            {
                m_no_improvement_count = 0;
            }

            if (m_currentLoss < m_best_loss)
            {
                m_best_loss = m_currentLoss;
            }
        }

        double BackPropagate(Matrix<float> x, Matrix<float> y,
            List<Matrix<float>> activations, List<Matrix<float>> activationsWork, List<Matrix<float>> deltas,
            List<Matrix<float>> coefGrads, List<Vector<float>> interceptGrads)
        {
            var n_samples = x.RowCount;

            ForwardPass(activations, activationsWork);

            var loss = (double)m_loss.Loss(y, activations[activations.Count - 1]);

            // Add L2 regularization term to loss
            if (m_l2reguralization != 0.0)
            {
                var sum = 0.0;
                foreach (var coef in m_coefs)
                {
                    sum += coef.ElementWiseMultiplicationSum(coef);
                }
                loss += (0.5 * m_l2reguralization) * sum / (double)n_samples;
            }

            var last = m_n_layers - 2;

            // clear deltas from last iteration
            deltas.ForEach(d => d.Clear());

            //The calculation of delta[last] here works with following
            //combinations of output activation and loss function:
            //sigmoid and binary cross entropy, softmax and categorical cross
            //entropy, and identity with squared loss            
            y.Subtract(activations[activations.Count - 1], deltas[last]);
            deltas[last].Multiply(-1f, deltas[last]);

            ComputLossGrad(last, n_samples, activations, activationsWork,
                deltas, coefGrads, interceptGrads);

            for (int i = m_n_layers - 1; i-- > 1; )
            {
                deltas[i].TransposeAndMultiply(m_coefs[i], deltas[i - 1]);

                activationsWork[i].Clear();
                m_hiddenActiviation.Derivative(activations[i], activationsWork[i]);
                deltas[i - 1].PointwiseMultiply(activationsWork[i], deltas[i - 1]);

                if (m_dropout && m_dropOutValues[i] > 0.0)
                {
                    deltas[i - 1].PointwiseMultiply(m_dropMasks[i], deltas[i - 1]);
                }

                ComputLossGrad(i - 1, n_samples, activations, activationsWork,
                    deltas, coefGrads, interceptGrads);
            }

            return loss;
        }

        void ComputLossGrad(int layer, int nSamples, List<Matrix<float>> activations, List<Matrix<float>> activationsWork,
            List<Matrix<float>> deltas, List<Matrix<float>> coefGrads, List<Vector<float>> interceptGrads)
        {
            // clear gradients from last iteration
            coefGrads[layer].Clear();
            interceptGrads[layer].Clear();

            if (m_dropout && m_dropOutValues[layer] > 0.0)
            {
                activationsWork[layer].Clear();
                activations[layer].PointwiseMultiply(m_dropMasks[layer], activationsWork[layer]);
                activationsWork[layer].TransposeThisAndMultiply(deltas[layer], coefGrads[layer]);
            }
            else
            {
                activations[layer].TransposeThisAndMultiply(deltas[layer], coefGrads[layer]);
            }

            var coef = m_coefs[layer];
            coefGrads[layer].MapIndexed((r, c, v) => (v + m_l2reguralization * coef[r, c]) / (float)nSamples, coefGrads[layer]);
            deltas[layer].ColumnWiseMean(interceptGrads[layer]);
        }

        void ForwardPass(List<Matrix<float>> activations, List<Matrix<float>> activationsWork)
        {
            for (int i = 0; i < m_n_layers - 1; i++)
            {
                // clear activations from previous interation
                activations[i + 1].Clear();

                if (m_dropout && m_dropOutValues[i] > 0.0)
                {
                    UpdateMask(i);
                    // clear work activations
                    activationsWork[i].Clear();

                    activations[i].PointwiseMultiply(m_dropMasks[i], activationsWork[i]);
                    activationsWork[i].Multiply(m_coefs[i], activations[i + 1]);
                }
                else
                {
                    activations[i].Multiply(m_coefs[i], activations[i + 1]);
                }

                activations[i + 1].AddRowWise(m_intercepts[i], activations[i + 1]);

                if ((i + 1) != m_n_layers - 1)
                {
                    m_hiddenActiviation.Activation(activations[i + 1]);
                }
            }

            // output activation
            m_outputActiviation.Activation(activations[m_n_layers - 1]);
        }

        void CopyBatchTargets(Matrix<float> targets, Matrix<float> batch, int[] indices)
        {
            var cols = targets.ColumnCount;
            var batchRow = 0;
            foreach (var row in indices)
            {
                for (int col = 0; col < cols; col++)
                {
                    batch[batchRow, col] = targets[row, col];
                }
                batchRow++;
            }
        }

        void CopyBatch(F64Matrix observations, Matrix<float> batch, int[] indices)
        {
            var cols = observations.GetNumberOfColumns();
            var batchRow = 0;
            foreach (var row in indices)
            {
                for (int col = 0; col < cols; col++)
                {
                    batch[batchRow, col] = (float)observations[row, col];
                }
                batchRow++;
            }
        }

        void Initialize(Matrix<float> targets, List<int> layerUnits)
        {
            m_n_iter = 0;
            m_t = 0;
            m_n_outputs = targets.ColumnCount;
            m_n_layers = layerUnits.Count;

            m_coefs = new List<Matrix<float>>();
            m_intercepts = new List<Vector<float>>();

            for (int i = 0; i < m_n_layers - 1; i++)
            {
                var pair = InitCoef(layerUnits[i], layerUnits[i + 1]);
                m_coefs.Add(pair.Item1);
                m_intercepts.Add(pair.Item2);
            }

            m_no_improvement_count = 0;
            m_best_loss = double.MaxValue;
        }

        Tuple<Matrix<float>, Vector<float>> InitCoef(int fanIn, int fanOut)
        {
            var bound = m_hiddenActiviation.InitializationBound(fanIn, fanOut);
            var distribution = new ContinuousUniform(-bound, bound, new Random(m_random.Next()));

            var coef = Matrix<float>.Build.Random(fanIn, fanOut, distribution);
            var intercept = Vector<float>.Build.Random(fanOut, distribution);

            return Tuple.Create(coef, intercept);
        }

        void UpdateMask(int layer)
        {
            m_dropArrays[layer].Shuffle(m_dropoutRandom);
        }

        float DecideDropOut(int layer)
        {
            var dropOutScale = 1.0 / (1.0 - m_dropOutValues[layer]);
            return (float)(m_dropoutRandom.NextDouble() > m_dropOutValues[layer] ? dropOutScale * 1.0 : 0.0);
        }
    }
}
