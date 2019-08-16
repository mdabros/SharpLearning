using System;

namespace SharpLearning.Optimization.Transforms
{
    /// <summary>
    /// Return a transform from predefined selections.
    /// </summary>
    public static class TransformFactory
    {
        /// <summary>
        /// Return a transform from predefined selections.
        /// </summary>
        /// <param name="transform"></param>
        /// <returns></returns>
        public static ITransform Create(Transform transform)
        {
            switch (transform)
            {
                case Transform.Linear:
                    return new LinearTransform();
                case Transform.Log10:
                    return new Log10Transform();
                case Transform.ExponentialAverage:
                    return new ExponentialAverageTransform();
                default:
                    throw new ArgumentException("Unsupported transform: " + transform);
            }
        }
    }
}
