namespace SharpLearning.Neural.Optimizers
{
    /// <summary>
    /// Optimization methods for neural net learning.
    /// </summary>
    public enum OptimizerMethod
    {
        /// <summary>
        /// Stochastic gradient descent. 
        /// Recommended learning rate: 0.01.
        /// </summary>
        Sgd,

        /// <summary>
        /// Adam (Adaptive Moment Estimation) is another method that computes adaptive learning rates for each parameter. 
        /// In addition to storing an exponentially decaying average of past squared gradients vtvt like Adadelta, 
        /// Adam also keeps an exponentially decaying average of past gradients, similar to momentum.
        /// Essentially Adam is RMSProp with momentum.
        /// https://arxiv.org/pdf/1412.6980.pdf.
        /// Recommended learning rate: 0.001.
        /// </summary>
        Adam,

        /// <summary>
        /// AdaMax is a variant of Adam based on the infinity norm. The algorithm is descriped in the Adam paper section 7:
        /// https://arxiv.org/pdf/1412.6980.pdf.
        /// Recommended learning rate: 0.002.
        /// </summary>
        AdaMax,

        /// <summary>
        /// Much like Adam is essentially RMSprop with momentum, Nadam is RMSprop with Nesterov momentum:
        /// http://cs229.stanford.edu/proj2015/054_report.pdf.
        /// Recommended learning rate: 0.002
        /// </summary>
        Nadam,

        /// <summary>
        /// Adagrad adapts the learning rate to each parameter, performing larger updates for infrequent and smaller updates for frequent parameters. 
        /// For this reason, it is well-suited for dealing with sparse data:
        /// https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad.
        /// Recommended learning rate: 0.01.
        /// </summary>
        Adagrad,

        /// <summary>
        /// Adadelta is an extension of Adagrad that seeks to reduce its aggressive, 
        /// monotonically decreasing learning rate. 
        /// Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size w.
        /// https://arxiv.org/pdf/1212.5701v1.pdf.
        /// Recommended learning rate: 1.0.
        /// </summary>
        Adadelta,

        /// <summary>
        /// Stochastic gradient descent but with Nesterov's accelareted gradient.
        /// Recommended learning rate: 0.01.
        /// </summary>
        Netsterov,

        /// <summary>
        /// RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton.
        /// RMSprop and Adadelta have both been developed independently around the same time stemming from the need to resolve Adagrad's radically diminishing learning rates.
        /// Recommended learning rate: 0.001.
        /// </summary>
        RMSProp
    }
}
