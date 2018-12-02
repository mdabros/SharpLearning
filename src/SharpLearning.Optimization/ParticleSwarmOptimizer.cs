using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Containers.Arithmetic;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Particle Swarm optimizer (PSO). PSO is initialized with a group of random particles
    /// and then searches for optima by updating generations. In every iteration, each particle is updated by following two "best" values. 
    /// The first one is the best solution found by the specific particle so far. 
    /// The other "best" value is the global best value obtained by any particle in the population so far.
    /// http://www.swarmintelligence.org/tutorials.php
    /// https://en.wikipedia.org/wiki/Particle_swarm_optimization
    /// </summary>
    public sealed class ParticleSwarmOptimizer : IOptimizer
    {
        readonly MinMaxParameter[] m_parameters;
        readonly int m_maxIterations;
        readonly int m_numberOfParticles;
        readonly double m_c1;
        readonly double m_c2;
        readonly Random m_random;
        readonly IParameterSampler m_sampler;

        /// <summary>
        /// Particle Swarm optimizer (PSO). PSO is initialized with a group of random particles
        /// and then searches for optima by updating generations. In every iteration, each particle is updated by following two "best" values. 
        /// The first one is the best solution found by the specific particle so far. 
        /// The other "best" value is the global best value obtained by any particle in the population so far.
        /// </summary>
        /// <param name="parameters">A list of parameter bounds for each optimization parameter</param>
        /// <param name="maxIterations">Maximum number of iterations. MaxIteration * numberOfParticles = totalFunctionEvaluations</param>
        /// <param name="numberOfParticles">The number of particles to use (default is 10). MaxIteration * numberOfParticles = totalFunctionEvaluations</param>
        /// <param name="c1">Learning factor weigting local particle best solution. (default is 2)</param>
        /// <param name="c2">Learning factor weigting global best solution. (default is 2)</param>
        /// <param name="seed">Seed for the random initialization and velocity corrections</param>
        public ParticleSwarmOptimizer(MinMaxParameter[] parameters, int maxIterations, int numberOfParticles = 10, double c1 = 2, double c2 = 2, int seed = 42)
        {
            if (parameters == null) { throw new ArgumentNullException("parameters"); }
            if (maxIterations <= 0) { throw new ArgumentNullException("maxIterations must be at least 1"); }
            if (numberOfParticles < 1) { throw new ArgumentNullException("numberOfParticles must be at least 1"); }

            m_parameters = parameters;
            m_maxIterations = maxIterations;
            m_numberOfParticles = numberOfParticles;
            m_c1 = c1;
            m_c2 = c2;
            
            m_random = new Random(seed);
            
            // Use member to seed the random uniform sampler.
            m_sampler = new RandomUniform(m_random.Next());
        }

        /// <summary>
        /// Optimization using swarm optimization.
        /// Returns the result which best minimises the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult OptimizeBest(Func<double[], OptimizerResult> functionToMinimize)
        {
            return Optimize(functionToMinimize).First();
        }

        /// <summary>
        /// Optimization using swarm optimization.
        /// Returns the final results ordered from best to worst (minimized).
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            var particles = new double[m_numberOfParticles][];

            var particleVelocities = Enumerable.Range(0, m_numberOfParticles)
                .Select(p => new double[m_parameters.Length])
                .ToArray();

            // initialize max and min velocities
            var maxParticleVelocities = new double[m_parameters.Length];
            var minParticleVelocities = new double[m_parameters.Length];
            for (int i = 0; i < m_parameters.Length; i++)
            {
                maxParticleVelocities[i] = Math.Abs(m_parameters[i].Max - m_parameters[i].Min);
                minParticleVelocities[i] = -maxParticleVelocities[i];
            }

            // initialize max and min parameter bounds
            var maxParameters = new double[m_parameters.Length];
            var minParameters = new double[m_parameters.Length];
            for (int i = 0; i < m_parameters.Length; i++)
            {
                maxParameters[i] = m_parameters[i].Max;
                minParameters[i] = m_parameters[i].Min;
            }

            var pBest = Enumerable.Range(0, m_numberOfParticles)
                .Select(p => new double[m_parameters.Length])
                .ToArray();

            var pBestScores = Enumerable.Range(0, m_numberOfParticles)
                .Select(p => double.MaxValue)
                .ToArray();

            var gBest = new OptimizerResult(new double[m_parameters.Length], double.MaxValue);
            
            // random initialize particles
            for (int i = 0; i < m_numberOfParticles; i++)
            {
                particles[i] = CreateParticle();
            }

            // iterate for find best
            for (int iterations = 0; iterations < m_maxIterations; iterations++)
            {
                for (int i = 0; i < m_numberOfParticles; i++)
                {
                    var result = functionToMinimize(particles[i]);
                    if(result.Error < pBestScores[i])
                    {
                        pBest[i] = result.ParameterSet;
                        pBestScores[i] = result.Error;
                    }

                    if(result.Error < gBest.Error)
                    {
                        gBest = new OptimizerResult(result.ParameterSet.ToArray(), result.Error);
                        //Trace.WriteLine(gBest.Error);
                    }
                }

                for (int i = 0; i < m_numberOfParticles; i++)
                {
                    //v[] = v[] + c1 * rand() * (pbest[] - present[]) + c2 * rand() * (gbest[] - present[])
                    particleVelocities[i] = particleVelocities[i].Add(pBest[i].Subtract(particles[i]).Multiply(m_c1 * m_random.NextDouble())
                        .Add(gBest.ParameterSet.Subtract(particles[i]).Multiply(m_c2 * m_random.NextDouble())));

                    BoundCheck(particleVelocities[i], maxParticleVelocities, minParticleVelocities);

                    //present[] = persent[] + v[]
                    particles[i] = particles[i].Add(particleVelocities[i]);
                    BoundCheck(particles[i], maxParameters, minParameters);

                }
            }

            var results = new List<OptimizerResult>();
            for (int i = 0; i < m_numberOfParticles; i++)
            {
                results.Add(new OptimizerResult(pBest[i], pBestScores[i]));
            }                            

            return results.Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).ToArray();
        }

        void BoundCheck(double[] newValues, double[] maxValues, double[] minValues)
        {
            for (int i = 0; i < newValues.Length; i++)
            {
                newValues[i] = Math.Max(minValues[i], Math.Min(newValues[i], maxValues[i]));
            }
        }

        double[] CreateParticle()
        {
            var newPoint = new double[m_parameters.Length];

            for (int i = 0; i < m_parameters.Length; i++)
            {
                var parameter = m_parameters[i];
                newPoint[i] = parameter.SampleValue(m_sampler);
            }

            return newPoint;
        }
    }
}
