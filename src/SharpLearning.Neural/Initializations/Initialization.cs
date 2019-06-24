namespace SharpLearning.Neural.Initializations
{
    /// <summary>
    /// Specifies the different types of initialization.
    /// </summary>
    public enum Initialization
    {
        /// <summary>
        /// Glorot initialization using uniform distribution, based on paper:
        /// http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        /// </summary>
        GlorotUniform,

        /// <summary>
        /// He initialization using uniform distribution, based on paper:
        /// https://arxiv.org/pdf/1502.01852.pdf
        /// </summary>
        HeUniform,

        /// <summary>
        /// Glorot initialization using normal distribution, based on paper:
        /// http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        /// </summary>
        GlorotNormal,

        /// <summary>
        /// He initialization using normal distribution, based on paper:
        /// https://arxiv.org/pdf/1502.01852.pdf
        /// </summary>
        HeNormal,
    }
}
