using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Models;
using SharpLearning.Neural.Optimizers;
using SharpLearning.Neural.TargetEncoders;

namespace SharpLearning.Neural.Learners
{
    /// <summary>
    /// RegressionNeuralNet learner utilizing the nesterov momentum method for stochastic optimization.
    /// https://en.wikipedia.org/wiki/Gradient_descent
    /// </summary>
    public sealed class RegressionMomentumNeuralNetLearner : IIndexedLearner<double>, ILearner<double>
    {
        readonly NeuralNetLearner m_learner;

        /// <summary>
        /// RegressionNeuralNet learner utilizing the nesterov momentum method for stochastic optimization.
        /// https://en.wikipedia.org/wiki/Gradient_descent
        /// </summary>
        public RegressionMomentumNeuralNetLearner()
            : this(new HiddenLayer[] { HiddenLayer.New(100) }, new ReluActivation(), new SquaredLoss())
        {
        }

        /// <summary>
        /// RegressionNeuralNet learner utilizing the nesterov momentum method for stochastic optimization.
        /// https://en.wikipedia.org/wiki/Gradient_descent
        /// </summary>
        /// <param name="hiddenLayers">Hidden layers. The layers is initializes in the order they appear in the array</param>
        public RegressionMomentumNeuralNetLearner(HiddenLayer[] hiddenLayers)
            : this(hiddenLayers, new ReluActivation(), new SquaredLoss())
        {
        }

        /// <summary>
        /// RegressionNeuralNet learner utilizing the nesterov momentum method for stochastic optimization.
        /// https://en.wikipedia.org/wiki/Gradient_descent
        /// </summary>
        /// <param name="hiddenLayers">Hidden layers. The layers is initializes in the order they appear in the array.
        /// (default is single layer with 100 units)</param>
        /// <param name="activiation">The type of activation used in the hidden layers (Default is Rectifier linear units)</param>
        public RegressionMomentumNeuralNetLearner(HiddenLayer[] hiddenLayers, IActivation activiation)
            : this(hiddenLayers, activiation, new SquaredLoss())
        {
        }

        /// <summary>
        /// RegressionNeuralNet learner utilizing the nesterov momentum method for stochastic optimization.
        /// https://en.wikipedia.org/wiki/Gradient_descent
        /// </summary>
        /// <param name="hiddenLayers">Hidden layers. The layers is initializes in the order they appear in the array 
        /// (default is single layer with 100 units)</param>
        /// <param name="activiation">The type of activation used in the hidden layers (Default is Rectifier linear units)</param>
        /// <param name="loss">Loss to minimize (Default is square loss)</param>
        /// <param name="l2regularization">L2 reguralization term. (Default is 0.0001)</param>
        /// <param name="inputDropOut">Input dropout percentage. The percentage of units randomly omitted during training.
        /// This is a reguralizatin methods for reducing overfitting. Recommended value is 0.1 and range should be between 0.0 and 0.3. Default is (0.0)
        /// https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf</param>
        /// <param name="batchSize">Batch size for mini-batch stochastic gradient descent.(Default is 200)</param>
        /// <param name="learningRateSchedule">Learning rate Schedule. This specifies the schedule for updating the weights.(Defalt is "Constant").
        /// "Constant" uses the initial learning rate in all iterations.
        /// "InvScale" gradueally decreases the learning rate in each iteration using an inverse scaling
        /// exponent of powert.
        /// "Adaptive" reduces the learning rate by dividing by 5 each time the training loss has not gone down 2 times in a row.</param>
        /// <param name="learningRateInitial">Initial learning rate. Controls the step size when updating the weights. (Default is 0.001)</param>
        /// <param name="powerT">The exponent of inverse scaling. This is only used when "InvScale" learning rate types is selected. (Default is 0.5)</param>
        /// <param name="maxIterations">The maximum number of iterations before termination. (Default is 100)</param>
        /// <param name="shuffle">Decides if the observations should be shuffled between each iteration. (Default is true)</param>
        /// <param name="seed">Seed for random initialization of weights. (defualt is 42)</param>
        /// <param name="tol">Tolerence for the optimization. If the training loss has not improved be tol 
        /// in two consequitive iterations. The optimization terminates. (Default is 0.0001)</param>
        /// <param name="momentum">Momentum for gradient update. Should be between 0 and 1. (Defualt is 0.9)</param>
        /// <param name="useNesterovsMomentum">Wether to use nesterov momentum.</param>
        public RegressionMomentumNeuralNetLearner(HiddenLayer[] hiddenLayers, IActivation activiation, ILoss loss, int maxIterations = 200, double learningRateInitial = 0.001,
            int batchSize = 100, double l2regularization = 0.0001, double inputDropOut = 0.0, LearningRateSchedule learningRateSchedule = LearningRateSchedule.Constant,
            double powerT = 0.5, bool shuffle = true, int seed = 42, double tol = 1e-4f, double momentum = 0.9f, bool useNesterovsMomentum = true)
        {
            m_learner = new NeuralNetLearner(hiddenLayers, activiation, loss, new CopyTargetEncoder(), new IdentityActivation(),
                new SGDNeuralNetOptimizer(learningRateInitial, learningRateSchedule, momentum, useNesterovsMomentum, powerT), maxIterations,
                batchSize, l2regularization, inputDropOut, shuffle, seed, tol);
        }

        /// <summary>
        /// Learns a regression neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public RegressionNeuralNetModel Learn(F64Matrix observations, double[] targets)
        {
            return new RegressionNeuralNetModel(m_learner.Learn(observations, targets));
        }

        /// <summary>
        /// Learns a regression neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public RegressionNeuralNetModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return new RegressionNeuralNetModel(m_learner.Learn(observations, targets, indices));
        }

        /// <summary>
        /// Learns a regression neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        /// <summary>
        /// Learns a regression neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }
    }
}
