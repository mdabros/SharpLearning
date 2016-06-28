using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Models;
using SharpLearning.Neural.Optimizers;
using SharpLearning.Neural.TargetEncoders;
using System.Linq;

namespace SharpLearning.Neural.Learners
{
    /// <summary>
    /// ClassificationNeuralNet learner utilizing the Adam method for stochastic optimization. 
    /// is well suited for problems that are large in terms of data and/or parameters.
    /// The method is also appropriate for non-stationary objectives and problems with
    /// very noisy and/or sparse gradients. https://arxiv.org/pdf/1412.6980.pdf
    /// </summary>
    public class ClassificationAdamNeuralNetLearner : IIndexedLearner<double>, IIndexedLearner<ProbabilityPrediction>,
        ILearner<double>, ILearner<ProbabilityPrediction>
    {
        readonly NeuralNetLearner m_learner;


        /// <summary>
        /// ClassificationNeuralNet learner utilizing the Adam method for stochastic optimization. 
        /// is well suited for problems that are large in terms of data and/or parameters.
        /// The method is also appropriate for non-stationary objectives and problems with
        /// very noisy and/or sparse gradients. https://arxiv.org/pdf/1412.6980.pdf
        /// </summary>
        public ClassificationAdamNeuralNetLearner()
            : this(new HiddenLayer[] { HiddenLayer.New(100) }, new ReluActivation(), new LogLoss())
        {
        }

        /// <summary>
        /// ClassificationNeuralNet learner utilizing the Adam method for stochastic optimization. 
        /// is well suited for problems that are large in terms of data and/or parameters.
        /// The method is also appropriate for non-stationary objectives and problems with
        /// very noisy and/or sparse gradients. https://arxiv.org/pdf/1412.6980.pdf
        /// </summary>
        /// <param name="hiddenLayers">Hidden layers. The layers is initializes in the order they appear in the array</param>
        public ClassificationAdamNeuralNetLearner(HiddenLayer[] hiddenLayers)
            : this(hiddenLayers, new ReluActivation(), new LogLoss())
        {
        }

        /// <summary>
        /// ClassificationNeuralNet learner utilizing the Adam method for stochastic optimization. 
        /// is well suited for problems that are large in terms of data and/or parameters.
        /// The method is also appropriate for non-stationary objectives and problems with
        /// very noisy and/or sparse gradients. https://arxiv.org/pdf/1412.6980.pdf
        /// </summary>
        /// <param name="hiddenLayers">Hidden layers. The layers is initializes in the order they appear in the array.
        /// (default is single layer with 100 units)</param>
        /// <param name="activiation">The type of activation used in the hidden layers (Default is Rectifier linear units)</param>
        public ClassificationAdamNeuralNetLearner(HiddenLayer[] hiddenLayers, IActivation activiation)
            : this(hiddenLayers, activiation, new LogLoss())
        {
        }

        /// <summary>
        /// ClassificationNeuralNet learner utilizing the Adam method for stochastic optimization. 
        /// is well suited for problems that are large in terms of data and/or parameters.
        /// The method is also appropriate for non-stationary objectives and problems with
        /// very noisy and/or sparse gradients. https://arxiv.org/pdf/1412.6980.pdf
        /// </summary>
        /// <param name="hiddenLayers">Hidden layers. The layers is initializes in the order they appear in the array 
        /// (default is single layer with 100 units)</param>
        /// <param name="activiation">The type of activation used in the hidden layers (Default is Rectifier linear units)</param>
        /// <param name="loss">Loss to minimize (Default is LogLoss)</param>
        /// <param name="maxIterations">The maximum number of iterations before termination. (Default is 100)</param>
        /// <param name="learningRateInitial">Initial learning rate. Controls the step size when updating the weights. (Default is 0.001)</param>
        /// <param name="batchSize">Batch size for mini-batch stochastic gradient descent.(Default is 200)</param>
        /// <param name="l2regularization">L2 reguralization term. (Default is 0.0001)</param>
        /// <param name="inputDropOut">Input dropout percentage. The percentage of units randomly omitted during training.
        /// This is a reguralizatin methods for reducing overfitting. Recommended value is 0.1 and range should be between 0.0 and 0.3. Default is (0.0)
        /// https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf</param>
        /// <param name="beta1">Exponential decay rate for estimates of first moment vector, should be in range 0 to 1</param>
        /// <param name="beta2">Exponential decay rate for estimates of second moment vector, should be in range 0 to 1</param>
        /// <param name="shuffle">Decides if the observations should be shuffled between each iteration. (Default is true)</param>
        /// <param name="seed">Seed for random initialization of weights. (defualt is 42)</param>
        /// <param name="tol">Tolerence for the optimization. If the training loss has not improved be tol 
        /// in two consequitive iterations. The optimization terminates. (Default is 0.0001)</param>
        public ClassificationAdamNeuralNetLearner(HiddenLayer[] hiddenLayers, IActivation activiation, ILoss loss, int maxIterations = 200, double learningRateInitial = 0.001,
            int batchSize = 100, double l2regularization = 0.0001, double inputDropOut = 0.0, double beta1 = 0.9, double beta2 = 0.999, bool shuffle = true, int seed = 42, double tol = 1e-4)
        {
            m_learner = new NeuralNetLearner(hiddenLayers, activiation, loss, new OneOfNTargetEncoder(), new SoftMaxActivation(),
                new AdamNeuralNetOptimizer(learningRateInitial, beta1, beta2), maxIterations, 
                batchSize, l2regularization, inputDropOut, shuffle, seed, tol);
        }

        /// <summary>
        /// Learns a classification neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public ClassificationNeuralNetModel Learn(F64Matrix observations, double[] targets)
        {
            var targetNames = GetOrderedTargetNames(targets);

            return new ClassificationNeuralNetModel(m_learner.Learn(observations, targets), targetNames);
        }

        /// <summary>
        /// Learns a classification neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public ClassificationNeuralNetModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            var targetNames = GetOrderedTargetNames(targets);

            return new ClassificationNeuralNetModel(m_learner.Learn(observations, targets, indices), targetNames);
        }

        /// <summary>
        /// Learns a classification neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        /// <summary>
        /// Learns a classification neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<ProbabilityPrediction> IIndexedLearner<ProbabilityPrediction>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }

        IPredictorModel<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Learns a classification neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<ProbabilityPrediction> ILearner<ProbabilityPrediction>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        double[] GetOrderedTargetNames(double[] targets)
        {
            return targets.Distinct().OrderBy(v => v).ToArray();
        }

    }
}
