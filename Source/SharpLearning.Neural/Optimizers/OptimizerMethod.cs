namespace SharpLearning.Neural.Optimizers
{
    /// <summary>
    /// Optimization methods for neural net learning.
    /// </summary>
    public enum OptimizerMethod
    {
        /// <summary>
        /// Stochastic gradient descent
        /// </summary>
        Sgd,

        /// <summary>
        /// Adam is a method for stochastic optimization:
        /// https://arxiv.org/pdf/1412.6980.pdf
        /// </summary>
        Adam,

        /// <summary>
        /// Adagrad is a method for stochastic optimization:
        /// https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad
        /// </summary>
        Adagrad,

        /// <summary>
        /// Adadelta: An adaptive learning rate method:
        /// https://arxiv.org/pdf/1212.5701v1.pdf
        /// </summary>
        Adadelta,

        /// <summary>
        /// Windowgrad is adagrad but with a moving window weighted average.
        /// </summary>
        Windowgrad,

        /// <summary>
        /// Stochastic gradient descent but with Nesterov's accelareted gradient.
        /// </summary>
        Netsterov
    }
}
