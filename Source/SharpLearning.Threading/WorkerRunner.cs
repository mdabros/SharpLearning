using System;
using System.Collections.Generic;
using System.Threading;

namespace SharpLearning.Threading
{
    /// <summary>
    /// Simple threading class for multithreaded work 
    /// </summary>
    /// <typeparam name="TResult"></typeparam>
    public sealed class WorkerRunner
    {
        readonly List<Action> m_workers; 

        /// <summary>
        /// Takes a list of worker actions as input. 
        /// The number of action corresponds to the number
        /// of threads started
        /// </summary>
        /// <param name="numberOfTreads"></param>
        public WorkerRunner(List<Action> workers)
        {
            if (workers == null) { throw new ArgumentNullException("workers"); }
            m_workers = workers;
        }

        /// <summary>
        /// Starts the threads and executes the actions
        /// </summary>
        /// <param name="tasks"></param>
        /// <param name="results"></param>
        public void Run()
        {
            var threads = new List<Thread>();
            foreach (var worker in m_workers)
            {
                Thread t = new Thread(new ThreadStart(() => worker()));
                t.Start();
                threads.Add(t);                
            }

            foreach (Thread t in threads) t.Join();
        }
    }
}
